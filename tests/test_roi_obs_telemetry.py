"""Tests for the A1 observability / billboard telemetry (FASTGS_ROI_OBS_TELEMETRY).

Covers the two new fast_utils primitives, the train.py guards that call them, and
the camera-load plumbing the flag turns on:
  * fast_utils.local_plane_normals      -- local PCA normals, k-guard, chunking
  * fast_utils.observability_telemetry  -- counts/purity/margin, BINARY support,
                                          M_hat -> c1/c2/h, rho, billboard pool
  * the train.py [ROI-OBS] guards (replicated in pure python + source-text checks)
  * loadCam's want_class_map condition (now also true for --roi_obs_telemetry)

Everything is CPU-only. render_fastgs is monkeypatched with fakes keyed by
(image_name, stencil rows), and utils.general_utils.build_rotation is monkeypatched
with a CPU mirror because the real one hardcodes device='cuda' (so
observability_telemetry itself is GPU-only in production); a CUDA-gated test
asserts the mirror is numerically identical to the real thing.
"""
import inspect
import math
import os
import re
from types import SimpleNamespace

import pytest
import torch

# CPU-only collection stubs, self-contained (never rely on sibling test files having
# collected first -- collection order is not guaranteed). Each stub is LOUD: reaching
# the real functionality fails the test instead of silently passing.
import sys
import types

try:
    import simple_knn  # noqa: F401
except ImportError:
    # Required only by scene.gaussian_model's distCUDA2, which these tests never
    # exercise.
    _simple_knn_stub = types.ModuleType("simple_knn")
    _simple_knn_c_stub = types.ModuleType("simple_knn._C")

    def _distCUDA2_stub(*_args, **_kwargs):
        raise RuntimeError("simple_knn._C.distCUDA2 stub called -- not built on this dev box")

    _simple_knn_c_stub.distCUDA2 = _distCUDA2_stub
    _simple_knn_stub._C = _simple_knn_c_stub
    sys.modules["simple_knn"] = _simple_knn_stub
    sys.modules["simple_knn._C"] = _simple_knn_c_stub

try:
    import fused_ssim  # noqa: F401
except ImportError:
    _fused_stub = types.ModuleType("fused_ssim")

    def _fused_ssim_stub(*_args, **_kwargs):
        raise RuntimeError("fused_ssim stub called -- compiled extension not built here")

    _fused_stub.fused_ssim = _fused_ssim_stub
    _fused_stub.FusedSSIMMap = _fused_ssim_stub
    sys.modules["fused_ssim"] = _fused_stub

try:
    import diff_gaussian_rasterization_fastgs  # noqa: F401
except ImportError:
    _stub = types.ModuleType("diff_gaussian_rasterization_fastgs")

    class _RasterizerUnavailable:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "diff_gaussian_rasterization_fastgs stubbed for CPU tests; a test "
                "reached the real rasterizer path, which these tests must not do.")

    _stub.GaussianRasterizationSettings = _RasterizerUnavailable
    _stub.GaussianRasterizer = _RasterizerUnavailable
    sys.modules["diff_gaussian_rasterization_fastgs"] = _stub

import scene.cameras  # noqa: F401 -- pre-existing utils.camera_utils <-> scene package
# circular import: entering via the scene package first initializes both sides cleanly.
from utils import camera_utils, fast_utils, general_utils

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# build_rotation: CPU mirror (the shipped one allocates on 'cuda')
# ---------------------------------------------------------------------------

_REAL_BUILD_ROTATION = general_utils.build_rotation


def _build_rotation_cpu(r):
    """CPU mirror of utils.general_utils.build_rotation (which hardcodes cuda)."""
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1]
                      + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])
    q = r / norm[:, None]
    R = torch.zeros((q.size(0), 3, 3), device=r.device, dtype=r.dtype)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


@pytest.fixture(autouse=True)
def _cpu_build_rotation(monkeypatch):
    """observability_telemetry does a deferred `from utils.general_utils import
    build_rotation`, so patching the module attribute reaches it at call time."""
    monkeypatch.setattr(general_utils, "build_rotation", _build_rotation_cpu)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="utils.general_utils.build_rotation hardcodes device='cuda'")
def test_cpu_rotation_mirror_matches_build_rotation():
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                          [0.5, 0.5, 0.5, 0.5],
                          [0.70710678, 0.70710678, 0.0, 0.0],
                          [2.0, -1.0, 0.5, 0.25]])
    ref = _REAL_BUILD_ROTATION(quats.cuda()).cpu()
    assert torch.allclose(_build_rotation_cpu(quats), ref, atol=1e-6)


# ---------------------------------------------------------------------------
# local_plane_normals
# ---------------------------------------------------------------------------

def _grid(nx, ny, z=0.0, dx=1.0):
    """nx*ny points on a z=const plane, spacing dx (row-major in x)."""
    pts = [[i * dx, j * dx, z] for i in range(nx) for j in range(ny)]
    return torch.tensor(pts, dtype=torch.float32)


def test_local_plane_normals_on_z_plane_are_plus_minus_z():
    # 25 coplanar points: the covariance is exactly rank 2, so the smallest
    # eigenvector is exactly +/-e_z (sign is arbitrary by construction).
    nrm = fast_utils.local_plane_normals(_grid(5, 5))
    assert nrm.shape == (25, 3) and nrm.dtype == torch.float32
    assert torch.allclose(nrm[:, 2].abs(), torch.ones(25), atol=1e-6)
    assert torch.allclose(nrm[:, 0:2], torch.zeros((25, 2)), atol=1e-6)
    assert torch.allclose(nrm.norm(dim=1), torch.ones(25), atol=1e-6)


def test_local_plane_normals_two_disjoint_parallel_planes():
    # Two 5x5 patches 10 apart; the widest in-patch distance is sqrt(32) = 5.66,
    # so every point's 16 nearest neighbours stay inside its own patch and both
    # patches must still report +/-e_z.
    pts = torch.cat([_grid(5, 5, z=0.0), _grid(5, 5, z=10.0)], dim=0)
    nrm = fast_utils.local_plane_normals(pts)
    assert nrm.shape == (50, 3)
    assert torch.allclose(nrm[:, 2].abs(), torch.ones(50), atol=1e-6)
    assert torch.allclose(nrm[:, 0:2], torch.zeros((50, 2)), atol=1e-6)


@pytest.mark.parametrize("m", [1, 15, 16])
def test_local_plane_normals_raises_when_m_le_k(m):
    pts = torch.zeros((m, 3), dtype=torch.float32)
    pts[:, 0] = torch.arange(m, dtype=torch.float32)
    with pytest.raises(ValueError, match=r"more than k=16"):
        fast_utils.local_plane_normals(pts)


def test_local_plane_normals_accepts_exactly_k_plus_one():
    # m == k + 1 is the smallest legal cloud: topk(k+1) takes the whole set.
    pts = torch.zeros((17, 3), dtype=torch.float32)
    pts[:, 0] = torch.arange(17, dtype=torch.float32)
    pts[:, 1] = torch.arange(17, dtype=torch.float32) % 3
    nrm = fast_utils.local_plane_normals(pts)
    assert nrm.shape == (17, 3)
    assert torch.allclose(nrm.norm(dim=1), torch.ones(17), atol=1e-6)


def test_local_plane_normals_smaller_k_is_honoured():
    with pytest.raises(ValueError, match=r"more than k=8"):
        fast_utils.local_plane_normals(torch.zeros((8, 3)), k=8)


def _two_orthogonal_patches():
    """30 points: 15 on the z=0 plane, 15 on a distant x=50 plane."""
    a = _grid(5, 3, z=0.0)                                        # 15 pts, z=0
    b = _grid(5, 3, z=0.0)
    b = torch.stack([torch.full((15,), 50.0), b[:, 0], b[:, 1]], dim=1)   # x=50
    return torch.cat([a, b], dim=0)


def test_local_plane_normals_chunking_is_equivalent():
    # 30 points, k=8: chunk=3 walks 10 chunks, chunk=10000 does it in one pass.
    pts = _two_orthogonal_patches()
    small = fast_utils.local_plane_normals(pts, k=8, chunk=3)
    big = fast_utils.local_plane_normals(pts, k=8, chunk=10000)
    assert torch.equal(small, big)
    # ... and the answer is the hand-known one: patch A normal +/-z, patch B +/-x
    assert torch.allclose(small[:15, 2].abs(), torch.ones(15), atol=1e-6)
    assert torch.allclose(small[:15, 0:2], torch.zeros((15, 2)), atol=1e-6)
    assert torch.allclose(small[15:, 0].abs(), torch.ones(15), atol=1e-6)
    assert torch.allclose(small[15:, 1:], torch.zeros((15, 2)), atol=1e-6)


# ---------------------------------------------------------------------------
# observability_telemetry: fixtures / fakes
# ---------------------------------------------------------------------------

ARGS = SimpleNamespace(mult=0.5)

LAB_ROWS = (2, 3)              # class 2 occupies rows 2-3 of _cm()
REM_ROWS = (0, 1, 4, 5)        # everything else
C0_ROWS = (5,)
C1_ROWS = (0, 1)
C3_ROWS = (4,)
ALL_ROWS = (0, 1, 2, 3, 4, 5)


def _cm():
    # 6x6: rows 0-1 class 1 (mount), rows 2-3 class 2 (label), row 4 class 3
    # (specimen), row 5 class 0 (background). Same helper as the sibling files.
    m = torch.zeros((6, 6), dtype=torch.uint8)
    m[0:2, :] = 1
    m[2:4, :] = 2
    m[4, :] = 3
    return m


def _rows_key(metric_map):
    """Row indices touched by a stencil -- every stencil used here is row-aligned,
    so the row tuple is an exact identity for the selected pixel set."""
    sel = metric_map > 0
    return tuple(int(r) for r in torch.nonzero(sel.any(dim=1)).flatten().tolist())


def _fake_render(calls, table, grad_flags=None):
    """render_fastgs stand-in whose per-gaussian counts are chosen per stencil.

    table maps (image_name, touched_rows) -> per-gaussian counts. An unexpected
    render raises KeyError, so the tests also pin down exactly WHICH stencils get
    rendered (and, by omission, that no extra remainder render happens).
    """
    def fake(cam, gaussians, pipe, bg, mult, get_flag=None, metric_map=None):
        assert get_flag is True
        # The CUDA kernel compares metric_map[pix] == 1 exactly; .int() must be int32.
        assert metric_map.dtype == torch.int32
        assert mult == ARGS.mult
        if grad_flags is not None:
            grad_flags.append(torch.is_grad_enabled())
        key = (cam.image_name, _rows_key(metric_map))
        calls.append(key)
        return {"accum_metric_counts": torch.tensor(table[key], dtype=torch.int32)}
    return fake


def _cam(name, center, class_map=None):
    if class_map is None:
        class_map = _cm()
    return SimpleNamespace(roi_class_map=class_map, image_name=name,
                           camera_center=torch.tensor(center, dtype=torch.float32))


IDENT_Q = [1.0, 0.0, 0.0, 0.0]
ROT_X90_Q = [0.70710678, 0.70710678, 0.0, 0.0]   # 90 deg about x: e_z -> -e_y
# scales are stored as logs; only their ORDER matters here. [1.0, 0.5, 0.0] means
# a1 = e_x (longest), a2 = e_y, a3 = e_z (shortest).
SCALE_XYZ = [1.0, 0.5, 0.0]
SCALE_YXZ = [0.5, 1.0, 0.0]                      # a1 = e_y, a2 = e_x, a3 = e_z


def _gauss(xyz, scaling=None, rotation=None):
    xyz = torch.tensor(xyz, dtype=torch.float32) if not torch.is_tensor(xyz) else xyz.float()
    n = xyz.shape[0]
    if scaling is None:
        scaling = [SCALE_XYZ] * n
    scaling = torch.tensor(scaling, dtype=torch.float32) if not torch.is_tensor(scaling) \
        else scaling.float()
    if rotation is None:
        rotation = [IDENT_Q] * n
    rotation = torch.tensor(rotation, dtype=torch.float32) if not torch.is_tensor(rotation) \
        else rotation.float()
    assert scaling.shape == (n, 3) and rotation.shape == (n, 4)
    return SimpleNamespace(get_xyz=xyz, _scaling=scaling, _rotation=rotation)


def _origin(n):
    return torch.zeros((n, 3), dtype=torch.float32)


# ---------------------------------------------------------------------------
# heartbeat mode (per_class=False): counts / purity
# ---------------------------------------------------------------------------

def test_heartbeat_two_renders_counts_and_purity(monkeypatch):
    calls = []
    table = {("A", LAB_ROWS): [20, 4, 0],
             ("A", REM_ROWS): [0, 6, 0]}
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_render(calls, table))
    tel = fast_utils.observability_telemetry(
        [_cam("A", (0.0, 0.0, 3.0))], _gauss(_origin(3)), None, None, ARGS, 2)
    assert sorted(calls) == sorted([("A", LAB_ROWS), ("A", REM_ROWS)])
    assert len(calls) == 2                              # label stencil + remainder
    # column 0 = label-stencil counts, column 1 = remainder counts
    assert torch.equal(tel["counts"], torch.tensor([[20.0, 0.0],
                                                    [4.0, 6.0],
                                                    [0.0, 0.0]]))
    # purity = lab / (lab + rem): 20/20, 4/10, and 0/0 -> 0 via clamp_min(1)
    assert torch.allclose(tel["purity"], torch.tensor([1.0, 0.4, 0.0]), atol=1e-6)
    assert tel["margin"] is None                        # heartbeat mode has no margin


def test_heartbeat_label_free_map_renders_remainder_only(monkeypatch):
    # A view whose class map contains no label pixels: one remainder render, counts
    # land in column 1, and the camera contributes no support (cam_lab stays None).
    calls = []
    cm = torch.ones((6, 6), dtype=torch.uint8)          # all class 1, no class 2
    table = {("A", ALL_ROWS): [7, 9]}
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_render(calls, table))
    tel = fast_utils.observability_telemetry(
        [_cam("A", (4.0, 0.0, 0.0), class_map=cm)], _gauss(_origin(2)),
        None, None, ARGS, 2)
    assert calls == [("A", ALL_ROWS)]
    assert torch.equal(tel["counts"], torch.tensor([[0.0, 7.0], [0.0, 9.0]]))
    assert torch.allclose(tel["purity"], torch.zeros(2))
    assert torch.equal(tel["support"], torch.zeros(2))
    assert torch.allclose(tel["rho"], torch.zeros(2))


# ---------------------------------------------------------------------------
# heartbeat mode: BINARY support weights (min_px)
# ---------------------------------------------------------------------------

def test_heartbeat_support_is_binary_at_min_px(monkeypatch):
    # min_px=16. Per (gaussian, camera): 15 label px contributes NOTHING to
    # support/rvec/M but still counts; 16 contributes exactly once.
    #   cam A (+x, label [15, 16]) -> supports g1 only
    #   cam B (+z, label [16, 15]) -> supports g0 only
    #   cam C (-y, label [1, 1])   -> supports nobody, whole camera short-circuits
    calls = []
    table = {("A", LAB_ROWS): [15, 16], ("A", REM_ROWS): [0, 0],
             ("B", LAB_ROWS): [16, 15], ("B", REM_ROWS): [0, 0],
             ("C", LAB_ROWS): [1, 1], ("C", REM_ROWS): [0, 0]}
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_render(calls, table))
    cams = [_cam("A", (1.0, 0.0, 0.0)), _cam("B", (0.0, 0.0, 1.0)),
            _cam("C", (0.0, -100.0, 0.0))]
    tel = fast_utils.observability_telemetry(
        cams, _gauss(_origin(2)), None, None, ARGS, 2, min_px=16)
    assert len(calls) == 6                              # all three cams still render
    # counts keep every pixel, including the sub-threshold ones: 15+16+1 = 32
    assert torch.equal(tel["counts"][:, 0], torch.tensor([32.0, 32.0]))
    assert torch.equal(tel["support"], torch.tensor([1.0, 1.0]))
    # one supporting camera each -> rho is exactly 1
    assert torch.allclose(tel["rho"], torch.tensor([1.0, 1.0]), atol=1e-6)
    # M_hat is diag(d d^T) of the single supporting camera; a1 = e_x, a2 = e_y:
    #   g0 supported by cam B (+z) -> M_hat = diag(0,0,1) -> c1 = 0, c2 = 0
    #   g1 supported by cam A (+x) -> M_hat = diag(1,0,0) -> c1 = 1, c2 = 0
    # c2 == 0 for both is what proves cam C (-y) never entered M.
    assert torch.allclose(tel["c1"], torch.tensor([0.0, 1.0]), atol=1e-6)
    assert torch.allclose(tel["c2"], torch.tensor([0.0, 0.0]), atol=1e-6)
    assert torch.allclose(tel["h"], torch.tensor([0.0, 1.0]), atol=1e-6)


def test_heartbeat_min_px_is_inclusive_and_parameterised(monkeypatch):
    # The same 15-count camera that is rejected at min_px=16 is accepted at 15.
    calls = []
    table = {("A", LAB_ROWS): [15], ("A", REM_ROWS): [0]}
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_render(calls, table))
    for min_px, expected in [(16, 0.0), (15, 1.0)]:
        tel = fast_utils.observability_telemetry(
            [_cam("A", (2.0, 0.0, 0.0))], _gauss(_origin(1)), None, None, ARGS, 2,
            min_px=min_px)
        assert float(tel["support"][0]) == expected


# ---------------------------------------------------------------------------
# heartbeat mode: M_hat -> c1 / c2 / h, and rho
# ---------------------------------------------------------------------------

def test_heartbeat_mhat_and_h_two_orthogonal_cameras(monkeypatch):
    # cams at +x and +z, both gaussians at the origin, both above min_px:
    #   d_A = e_x, d_B = e_z -> Msum = diag(1,0,1), support = 2
    #   M_hat = diag(0.5, 0, 0.5)
    # g0 scales [1.0, 0.5, 0.0]: a1 = e_x, a2 = e_y -> c1 = 0.5, c2 = 0
    #                            -> h = sqrt(0.5) = 0.7071068
    # g1 scales [0.5, 1.0, 0.0]: a1 = e_y, a2 = e_x -> c1 = 0, c2 = 0.5
    #                            -> c1 - c2 < 0 -> h clamps to exactly 0
    calls = []
    table = {("A", LAB_ROWS): [20, 20], ("A", REM_ROWS): [0, 0],
             ("B", LAB_ROWS): [20, 20], ("B", REM_ROWS): [0, 0]}
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_render(calls, table))
    cams = [_cam("A", (2.0, 0.0, 0.0)), _cam("B", (0.0, 0.0, 5.0))]
    tel = fast_utils.observability_telemetry(
        cams, _gauss(_origin(2), scaling=[SCALE_XYZ, SCALE_YXZ]),
        None, None, ARGS, 2)
    assert torch.equal(tel["support"], torch.tensor([2.0, 2.0]))
    assert torch.allclose(tel["c1"], torch.tensor([0.5, 0.0]), atol=1e-6)
    assert torch.allclose(tel["c2"], torch.tensor([0.0, 0.5]), atol=1e-6)
    assert torch.allclose(tel["h"], torch.tensor([math.sqrt(0.5), 0.0]), atol=1e-6)
    assert float(tel["h"][1]) == 0.0                     # exactly zero, not small
    # rho = |(e_x + e_z) / 2| = sqrt(0.5)
    assert torch.allclose(tel["rho"], torch.full((2,), math.sqrt(0.5)), atol=1e-6)


def test_heartbeat_rho_zero_for_opposing_cameras(monkeypatch):
    # +x and -x with equal binary support: rvec cancels (rho ~ 0) while M_hat does
    # NOT (d d^T is sign-invariant) -> c1 = 1 along the long axis.
    calls = []
    table = {("A", LAB_ROWS): [20], ("A", REM_ROWS): [0],
             ("B", LAB_ROWS): [20], ("B", REM_ROWS): [0]}
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_render(calls, table))
    cams = [_cam("A", (3.0, 0.0, 0.0)), _cam("B", (-7.0, 0.0, 0.0))]
    tel = fast_utils.observability_telemetry(
        cams, _gauss(_origin(1)), None, None, ARGS, 2)
    assert float(tel["support"][0]) == 2.0
    assert float(tel["rho"][0]) == pytest.approx(0.0, abs=1e-7)
    assert float(tel["c1"][0]) == pytest.approx(1.0, abs=1e-6)
    assert float(tel["c2"][0]) == pytest.approx(0.0, abs=1e-6)
    assert float(tel["h"][0]) == pytest.approx(1.0, abs=1e-6)


def test_heartbeat_rho_is_one_for_a_single_camera(monkeypatch):
    calls = []
    table = {("A", LAB_ROWS): [20], ("A", REM_ROWS): [0]}
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_render(calls, table))
    tel = fast_utils.observability_telemetry(
        [_cam("A", (0.0, 3.0, 4.0))], _gauss(_origin(1)), None, None, ARGS, 2)
    assert float(tel["rho"][0]) == pytest.approx(1.0, abs=1e-6)
    # d = (0, 0.6, 0.8); a1 = e_x -> c1 = 0, a2 = e_y -> c2 = 0.36 -> h = 0
    assert float(tel["c1"][0]) == pytest.approx(0.0, abs=1e-6)
    assert float(tel["c2"][0]) == pytest.approx(0.36, abs=1e-6)
    assert float(tel["h"][0]) == 0.0


def test_heartbeat_unobserved_gaussian_has_zero_moments(monkeypatch):
    # Never rendered onto: support 0 -> M_hat and rvec stay zero (ws clamps to 1),
    # so rho/c1/c2/h are all 0 and the row is excluded from the pool.
    calls = []
    table = {("A", LAB_ROWS): [0], ("A", REM_ROWS): [0]}
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_render(calls, table))
    tel = fast_utils.observability_telemetry(
        [_cam("A", (2.0, 0.0, 0.0))], _gauss(_origin(1)), None, None, ARGS, 2)
    for key in ("support", "rho", "c1", "c2", "h"):
        assert float(tel[key][0]) == 0.0
    assert bool(tel["pool_mask"][0]) is False


def test_heartbeat_failopen_camera_contributes_nothing(monkeypatch):
    # roi_class_map None -> skipped before any render; its center would otherwise
    # drag rho/M toward -x.
    calls = []
    table = {("A", LAB_ROWS): [20], ("A", REM_ROWS): [0]}
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_render(calls, table))
    cams = [_cam("A", (0.0, 0.0, 4.0)),
            SimpleNamespace(roi_class_map=None, image_name="B",
                            camera_center=torch.tensor([-100.0, 0.0, 0.0]))]
    tel = fast_utils.observability_telemetry(
        cams, _gauss(_origin(1)), None, None, ARGS, 2)
    assert len(calls) == 2 and all(name == "A" for name, _rows in calls)
    assert float(tel["support"][0]) == 1.0
    assert float(tel["rho"][0]) == pytest.approx(1.0, abs=1e-6)
    # d = e_z: a1 = e_x -> c1 = 0 ; had cam B counted, c1 would be 0.5
    assert float(tel["c1"][0]) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# heartbeat mode: pool_mask and billboard
# ---------------------------------------------------------------------------

def test_pool_mask_requires_purity_half_and_support_one(monkeypatch):
    # pool = (purity >= 0.5) & (support >= 1). min_px = 16.
    #   g0 lab 16 rem 16 -> purity 0.50 exactly, support 1 -> IN (boundary)
    #   g1 lab 15 rem 15 -> purity 0.50 exactly, support 0 -> OUT (support only)
    #   g2 lab 20 rem 30 -> purity 0.40,         support 1 -> OUT (purity only)
    #   g3 lab  0 rem  0 -> purity 0.00,         support 0 -> OUT
    calls = []
    table = {("A", LAB_ROWS): [16, 15, 20, 0], ("A", REM_ROWS): [16, 15, 30, 0]}
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_render(calls, table))
    tel = fast_utils.observability_telemetry(
        [_cam("A", (2.0, 0.0, 0.0))], _gauss(_origin(4)), None, None, ARGS, 2)
    assert torch.allclose(tel["purity"], torch.tensor([0.5, 0.5, 0.4, 0.0]), atol=1e-7)
    assert torch.equal(tel["support"], torch.tensor([1.0, 0.0, 1.0, 0.0]))
    assert tel["pool_mask"].tolist() == [True, False, False, False]
    # pool of 1 is <= K_LOCAL, so no normals are computed at all
    assert torch.isnan(tel["billboard"]).all()


def _flat_pool_scenario(n_pool, extra_lab=(), extra_rem=(), extra_xyz=(),
                        rotation=None, scaling=None):
    """n_pool gaussians on the z=0 plane (all high-purity, all supported by one
    camera at +z) plus optional non-pool gaussians appended after them."""
    grid = _grid(5, 5)[:n_pool]
    xyz = torch.cat([grid, torch.tensor(list(extra_xyz), dtype=torch.float32).reshape(-1, 3)],
                    dim=0)
    n = xyz.shape[0]
    if scaling is None:
        scaling = [SCALE_XYZ] * n
    if rotation is None:
        rotation = [IDENT_Q] * n
    table = {("A", LAB_ROWS): [20] * n_pool + list(extra_lab),
             ("A", REM_ROWS): [0] * n_pool + list(extra_rem)}
    return _gauss(xyz, scaling=scaling, rotation=rotation), table


def test_billboard_is_one_for_flat_gaussians(monkeypatch):
    # 25 coplanar high-purity gaussians -> pool 25 > K_LOCAL -> normals = +/-e_z;
    # a3 = e_z (shortest axis) -> billboard = |e_z . +/-e_z| = 1.
    calls = []
    gaussians, table = _flat_pool_scenario(25)
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_render(calls, table))
    tel = fast_utils.observability_telemetry(
        [_cam("A", (0.0, 0.0, 10.0))], gaussians, None, None, ARGS, 2)
    assert tel["pool_mask"].all()
    assert torch.allclose(tel["billboard"], torch.ones(25), atol=1e-6)


def test_billboard_is_zero_for_an_edge_on_gaussian(monkeypatch):
    # Same cloud, but gaussian 12 is rotated 90 deg about x, putting its shortest
    # axis in the plane: a3 = R e_z = -e_y -> |a3 . n| = |-e_y . +/-e_z| = 0.
    calls = []
    rot = [IDENT_Q] * 25
    rot[12] = ROT_X90_Q
    gaussians, table = _flat_pool_scenario(25, rotation=rot)
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_render(calls, table))
    tel = fast_utils.observability_telemetry(
        [_cam("A", (0.0, 0.0, 10.0))], gaussians, None, None, ARGS, 2)
    bb = tel["billboard"]
    assert float(bb[12]) == pytest.approx(0.0, abs=1e-6)
    others = torch.cat([bb[:12], bb[13:]])
    assert torch.allclose(others, torch.ones(24), atol=1e-6)


def test_billboard_is_nan_outside_the_pool(monkeypatch):
    # Three excluded rows, one per exclusion reason; all must stay NaN and must
    # not join the kNN cloud.
    calls = []
    gaussians, table = _flat_pool_scenario(
        25,
        extra_lab=(0, 20, 15),      # purity 0 / purity 0.4 / support 0 (15 < 16)
        extra_rem=(10, 30, 0),
        extra_xyz=((100.0, 0.0, 0.0), (100.0, 0.0, 1.0), (100.0, 0.0, 2.0)))
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_render(calls, table))
    tel = fast_utils.observability_telemetry(
        [_cam("A", (0.0, 0.0, 10.0))], gaussians, None, None, ARGS, 2)
    assert tel["pool_mask"][:25].all()
    assert not tel["pool_mask"][25:].any()
    assert torch.allclose(tel["purity"][25:], torch.tensor([0.0, 0.4, 1.0]), atol=1e-6)
    assert torch.equal(tel["support"][25:], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.isnan(tel["billboard"][25:]).all()
    assert torch.isfinite(tel["billboard"][:25]).all()
    assert torch.allclose(tel["billboard"][:25], torch.ones(25), atol=1e-6)


def test_billboard_defined_just_above_k_local(monkeypatch):
    # 17 pool rows: pool_idx.numel() (17) > K_LOCAL (16) -> normals ARE computed.
    calls = []
    gaussians, table = _flat_pool_scenario(17)
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_render(calls, table))
    tel = fast_utils.observability_telemetry(
        [_cam("A", (0.0, 0.0, 10.0))], gaussians, None, None, ARGS, 2)
    assert int(tel["pool_mask"].sum()) == 17
    assert torch.isfinite(tel["billboard"]).all()
    assert torch.allclose(tel["billboard"], torch.ones(17), atol=1e-6)


def _pool_of_16(monkeypatch, calls):
    """16 coincident, fully-supported gaussians: pool == K_LOCAL, so billboard is
    NaN everywhere while purity/support/h stay well defined."""
    table = {("A", LAB_ROWS): [20] * 16, ("A", REM_ROWS): [0] * 16}
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_render(calls, table))
    return fast_utils.observability_telemetry(
        [_cam("A", (5.0, 0.0, 0.0))], _gauss(_origin(16)), None, None, ARGS, 2)


def test_billboard_all_nan_at_exactly_k_local(monkeypatch):
    calls = []
    tel = _pool_of_16(monkeypatch, calls)
    assert int(tel["pool_mask"].sum()) == 16
    assert torch.isnan(tel["billboard"]).all()          # no crash, just undefined
    # the rest of the statistics are unaffected by the missing normals
    assert torch.allclose(tel["purity"], torch.ones(16))
    assert torch.equal(tel["support"], torch.ones(16))
    assert torch.allclose(tel["h"], torch.ones(16), atol=1e-6)   # d = e_x, a1 = e_x


def _pool_stride_replica(pool_idx, cap):
    """Replica of the POOL_CAP subsample in observability_telemetry (fast_utils.py,
    inside observability_telemetry: `step = (pool_idx.numel() + POOL_CAP - 1) //
    POOL_CAP; pool_idx = pool_idx[::step]`). POOL_CAP is a function-local constant
    (60000) and is therefore unpatchable; reaching it for real would need >60k
    gaussians, so the arithmetic is pinned here and the constant is pinned by the
    source-text test below."""
    if pool_idx.numel() > cap:
        step = (pool_idx.numel() + cap - 1) // cap
        pool_idx = pool_idx[::step]
    return pool_idx


@pytest.mark.parametrize("m,cap,expected", [
    (10, 10, 10),      # at the cap: untouched
    (11, 10, 6),       # step 2 over 11 -> 0,2,4,6,8,10
    (25, 10, 9),       # step 3 over 25 -> 0,3,...,24
    (100, 10, 10),     # step 10 over 100 -> 0,10,...,90
    (121, 10, 10),     # step 13 over 121 -> 0,13,...,117
])
def test_pool_cap_stride_is_bounded_and_deterministic(m, cap, expected):
    pool_idx = torch.arange(m)
    out = _pool_stride_replica(pool_idx, cap)
    assert out.numel() <= cap
    assert int(out[0]) == 0                              # first pool row always kept
    assert torch.equal(out, _pool_stride_replica(pool_idx, cap))   # deterministic
    step = 1 if m <= cap else (m + cap - 1) // cap
    assert torch.equal(out, pool_idx[::step])
    assert out.numel() == expected


def test_pool_cap_constants_are_pinned_in_source():
    src = inspect.getsource(fast_utils.observability_telemetry)
    assert re.search(r"^\s*K_LOCAL = 16\b", src, re.M)
    assert re.search(r"^\s*POOL_CAP = 60000\b", src, re.M)
    # r4: the kNN pass is additionally gated by with_billboard (scale-reg v2
    # refreshes skip it when the plate term is unarmed)
    assert "if with_billboard and pool_idx.numel() > K_LOCAL:" in src
    assert "step = (pool_idx.numel() + POOL_CAP - 1) // POOL_CAP" in src
    assert "pool_idx = pool_idx[::step]" in src
    # normals are asked for with the same k the guard checks against
    assert "local_plane_normals(gaussians.get_xyz.detach()[pool_idx], k=K_LOCAL)" in src


# ---------------------------------------------------------------------------
# per_class=True dump mode
# ---------------------------------------------------------------------------

def test_dump_renders_once_per_present_class(monkeypatch):
    # _cm() contains classes 0,1,2,3; class 4 is absent -> 4 renders, no class-4
    # render and no remainder render (the ids partition the map).
    calls = []
    table = {("A", C0_ROWS): [0, 0, 1],
             ("A", C1_ROWS): [0, 10, 2],
             ("A", LAB_ROWS): [20, 4, 0],
             ("A", C3_ROWS): [0, 0, 3]}
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_render(calls, table))
    tel = fast_utils.observability_telemetry(
        [_cam("A", (2.0, 0.0, 0.0))], _gauss(_origin(3)), None, None, ARGS, 2,
        per_class=True)
    assert sorted(calls) == sorted(table.keys())
    assert len(calls) == 4
    # counts has num_classes columns; column 4 never rendered -> stays zero
    assert torch.equal(tel["counts"], torch.tensor([[0.0, 0.0, 20.0, 0.0, 0.0],
                                                    [0.0, 10.0, 4.0, 0.0, 0.0],
                                                    [1.0, 2.0, 0.0, 3.0, 0.0]]))
    # purity uses the LABEL column (2), not column 0
    #   g0: 20/20 = 1 ; g1: 4/14 ; g2: 0/6 = 0
    assert torch.allclose(tel["purity"],
                          torch.tensor([1.0, 4.0 / 14.0, 0.0]), atol=1e-6)
    # margin = (lab - max_other) / tot
    #   g0: (20 - 0)/20 = 1 ; g1: (4 - 10)/14 = -0.428571 (class 1 wins) ;
    #   g2: (0 - 3)/6 = -0.5 (class 3 wins)
    assert torch.allclose(tel["margin"],
                          torch.tensor([1.0, -6.0 / 14.0, -0.5]), atol=1e-6)
    # support comes from the label column only: 20 >= 16, 4 and 0 do not
    assert torch.equal(tel["support"], torch.tensor([1.0, 0.0, 0.0]))


def test_dump_margin_is_zero_when_label_ties_the_runner_up(monkeypatch):
    calls = []
    table = {("A", C0_ROWS): [0], ("A", C1_ROWS): [8],
             ("A", LAB_ROWS): [8], ("A", C3_ROWS): [0]}
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_render(calls, table))
    tel = fast_utils.observability_telemetry(
        [_cam("A", (2.0, 0.0, 0.0))], _gauss(_origin(1)), None, None, ARGS, 2,
        per_class=True)
    assert float(tel["purity"][0]) == pytest.approx(0.5, abs=1e-7)
    assert float(tel["margin"][0]) == pytest.approx(0.0, abs=1e-7)


def test_dump_absent_label_class_gives_no_support(monkeypatch):
    # A view whose map has no label pixels at all: other classes still count, but
    # cam_lab stays None so the camera adds nothing to support/M/rvec.
    calls = []
    cm = torch.ones((6, 6), dtype=torch.uint8)          # class 1 only
    table = {("A", ALL_ROWS): [5]}
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_render(calls, table))
    tel = fast_utils.observability_telemetry(
        [_cam("A", (2.0, 0.0, 0.0), class_map=cm)], _gauss(_origin(1)),
        None, None, ARGS, 2, per_class=True)
    assert calls == [("A", ALL_ROWS)]
    assert torch.equal(tel["counts"], torch.tensor([[0.0, 5.0, 0.0, 0.0, 0.0]]))
    assert float(tel["support"][0]) == 0.0
    assert float(tel["margin"][0]) == pytest.approx(-1.0, abs=1e-7)   # (0 - 5)/5


# ---------------------------------------------------------------------------
# dtypes / shapes / signature contract
# ---------------------------------------------------------------------------

def test_heartbeat_dtypes_and_shapes(monkeypatch):
    calls = []
    table = {("A", LAB_ROWS): [20, 1, 0], ("A", REM_ROWS): [0, 1, 0]}
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_render(calls, table))
    tel = fast_utils.observability_telemetry(
        [_cam("A", (2.0, 0.0, 0.0))], _gauss(_origin(3)), None, None, ARGS, 2)
    assert tel["counts"].dtype == torch.float32 and tel["counts"].shape == (3, 2)
    for key in ("purity", "support", "rho", "c1", "c2", "h", "billboard"):
        assert tel[key].dtype == torch.float32, key
        assert tel[key].shape == (3,), key
    assert tel["pool_mask"].dtype == torch.bool and tel["pool_mask"].shape == (3,)
    assert tel["margin"] is None
    assert set(tel) == {"counts", "purity", "margin", "support", "rho", "c1", "c2",
                        "h", "billboard", "pool_mask"}


def test_dump_dtypes_and_shapes(monkeypatch):
    calls = []
    table = {("A", C0_ROWS): [1, 0], ("A", C1_ROWS): [1, 0],
             ("A", LAB_ROWS): [20, 0], ("A", C3_ROWS): [1, 0]}
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_render(calls, table))
    tel = fast_utils.observability_telemetry(
        [_cam("A", (2.0, 0.0, 0.0))], _gauss(_origin(2)), None, None, ARGS, 2,
        per_class=True)
    assert tel["counts"].dtype == torch.float32 and tel["counts"].shape == (2, 5)
    for key in ("purity", "margin", "support", "rho", "c1", "c2", "h", "billboard"):
        assert tel[key].dtype == torch.float32, key
        assert tel[key].shape == (2,), key
    assert tel["pool_mask"].dtype == torch.bool


def test_dump_honours_num_classes(monkeypatch):
    calls = []
    table = {("A", C0_ROWS): [1], ("A", C1_ROWS): [1], ("A", LAB_ROWS): [20]}
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_render(calls, table))
    tel = fast_utils.observability_telemetry(
        [_cam("A", (2.0, 0.0, 0.0))], _gauss(_origin(1)), None, None, ARGS, 2,
        per_class=True, num_classes=3)
    assert tel["counts"].shape == (1, 3)
    assert len(calls) == 3                               # class 3 is out of range now


def test_signature_defaults_match_the_train_py_contract():
    sig = inspect.signature(fast_utils.observability_telemetry)
    assert sig.parameters["min_px"].default == 16
    assert sig.parameters["per_class"].default is False
    assert sig.parameters["num_classes"].default == 5
    # train.py calls it positionally up to label_class_id
    assert list(sig.parameters)[:6] == ["camlist", "gaussians", "pipe", "bg", "args",
                                        "label_class_id"]


def test_telemetry_runs_with_grad_disabled(monkeypatch):
    # @torch.no_grad() on observability_telemetry: the renders must not build a
    # graph, and nothing returned may carry requires_grad.
    calls, grad_flags = [], []
    table = {("A", LAB_ROWS): [20], ("A", REM_ROWS): [0]}
    monkeypatch.setattr(fast_utils, "render_fastgs",
                        _fake_render(calls, table, grad_flags=grad_flags))
    gaussians = _gauss(_origin(1))
    gaussians.get_xyz.requires_grad_(True)
    gaussians._scaling.requires_grad_(True)
    gaussians._rotation.requires_grad_(True)
    assert torch.is_grad_enabled()                       # caller context has grad on
    tel = fast_utils.observability_telemetry(
        [_cam("A", (2.0, 0.0, 0.0))], gaussians, None, None, ARGS, 2)
    assert grad_flags == [False, False]                   # both renders under no_grad
    assert torch.is_grad_enabled()                       # and the context is restored
    for key, val in tel.items():
        if torch.is_tensor(val):
            assert not val.requires_grad, key


# ---------------------------------------------------------------------------
# train.py [ROI-OBS] guards: pure-python replicas + source-text checks
# ---------------------------------------------------------------------------

def _train_lines():
    with open(os.path.join(_REPO_ROOT, "train.py"), "r", encoding="utf-8") as fh:
        return fh.read().splitlines()


def _indent(line):
    return len(line) - len(line.lstrip())


def _enclosing_headers(lines, idx):
    """Statement headers enclosing lines[idx], innermost first (dedent walk)."""
    cur = _indent(lines[idx])
    out = []
    for j in range(idx - 1, -1, -1):
        stripped = lines[j].strip()
        if not stripped or stripped.startswith("#"):
            continue
        ind = _indent(lines[j])
        if ind < cur:
            out.append(stripped)
            cur = ind
            if cur == 0:
                break
    return out


def _call_line_indices(lines):
    return [i for i, ln in enumerate(lines) if "observability_telemetry(" in ln]


def test_train_py_telemetry_calls_are_flag_guarded_and_failsoft():
    lines = _train_lines()
    idxs = _call_line_indices(lines)
    assert len(idxs) == 4, ("expected heartbeat + final dump (telemetry) plus the two "
                            "scale-reg-v2 observability refreshes (r4)")
    obs_sites, v2_sites = [], []
    for i in idxs:
        headers = _enclosing_headers(lines, i)
        if "if roi_obs_telemetry:" in headers:
            obs_sites.append(i)
            # read-only diagnostics: they must never kill a run
            assert "try:" in headers, lines[i]
        else:
            v2_sites.append(i)
            # r4 refreshes feed a LOSS term: they sit behind the v2 gate only
            # and must fail LOUD — a swallowed refresh failure would let
            # training continue on a stale/absent observability cache.
            assert "if scale_reg_v2_on:" in headers, lines[i]
            assert "try:" not in headers, lines[i]
    assert len(obs_sites) == 2 and len(v2_sites) == 2
    # every other mention is the from-import continuation line, not a call
    others = [i for i, ln in enumerate(lines)
              if "observability_telemetry" in ln and i not in idxs]
    assert len(others) == 1 and "observability_telemetry," in lines[others[0]]


def test_train_py_heartbeat_cadence_and_off_path(monkeypatch):
    # Replica of train.py's heartbeat guard. Flag off -> the counter never moves
    # and observability_telemetry is never reached (sentinel would raise).
    def boom(*_a, **_k):
        raise AssertionError("telemetry ran on the off-path")

    monkeypatch.setattr(fast_utils, "observability_telemetry", boom)
    monkeypatch.setattr(fast_utils, "render_fastgs", boom)

    def run(flag, n_events):
        obs_event_i = 0
        fired = []
        for ev in range(1, n_events + 1):
            if flag:
                obs_event_i += 1
                if obs_event_i % 10 == 1:
                    fired.append(ev)
        return obs_event_i, fired

    assert run(False, 25) == (0, [])
    # positive control: the same replica fires on densify events 1, 11, 21
    monkeypatch.setattr(fast_utils, "observability_telemetry",
                        lambda *_a, **_k: {"pool_mask": torch.zeros(1, dtype=torch.bool)})
    assert run(True, 25) == (25, [1, 11, 21])


def test_train_py_heartbeat_stat_fallbacks(monkeypatch):
    # Replica of the split guard: with a non-empty pool whose billboard is all NaN
    # (pool <= K_LOCAL), bb_frac falls back to -1.0 while h/support/purity stay real.
    calls = []
    tel = _pool_of_16(monkeypatch, calls)
    pool = tel["pool_mask"]
    n_pool = int(pool.sum().item())
    bb = tel["billboard"][pool]
    bb = bb[torch.isfinite(bb)]
    assert n_pool == 16 and bb.numel() == 0
    if n_pool:
        bb_frac = float((bb < 0.5).float().mean().item()) if bb.numel() else -1.0
        h_med = float(tel["h"][pool].median().item())
        sup_med = float(tel["support"][pool].median().item())
        pur_med = float(tel["purity"][pool].median().item())
    else:
        bb_frac = h_med = sup_med = pur_med = -1.0
    assert bb_frac == -1.0                       # would be nan without the guard
    assert h_med == pytest.approx(1.0, abs=1e-6)
    assert sup_med == 1.0 and pur_med == pytest.approx(1.0, abs=1e-6)
    # source shape: bb_frac gates on bb.numel(), the rest on n_pool alone
    lines = _train_lines()
    assert any("if bb.numel() else -1.0" in ln for ln in lines)
    hb = next(i for i, ln in enumerate(lines) if ln.strip() == "if n_pool:")
    assert "if roi_obs_telemetry:" in _enclosing_headers(lines, hb)


def test_train_py_heartbeat_empty_pool_fallback(monkeypatch):
    # The n_pool == 0 branch, driven by a real all-background telemetry result:
    # every statistic reports the -1.0 sentinel instead of a median over nothing
    # (torch.median on an empty tensor would raise).
    calls = []
    table = {("A", LAB_ROWS): [0, 0], ("A", REM_ROWS): [10, 12]}
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_render(calls, table))
    tel = fast_utils.observability_telemetry(
        [_cam("A", (2.0, 0.0, 0.0))], _gauss(_origin(2)), None, None, ARGS, 2)
    pool = tel["pool_mask"]
    n_pool = int(pool.sum().item())
    bb = tel["billboard"][pool]
    bb = bb[torch.isfinite(bb)]
    assert n_pool == 0 and bb.numel() == 0
    if n_pool:
        bb_frac = float((bb < 0.5).float().mean().item()) if bb.numel() else -1.0
        h_med = sup_med = pur_med = float(tel["h"][pool].median().item())
    else:
        bb_frac = h_med = sup_med = pur_med = -1.0
    assert [bb_frac, h_med, sup_med, pur_med] == [-1.0, -1.0, -1.0, -1.0]
    assert torch.isnan(tel["billboard"]).all()


def test_train_py_obs_coverage_check_rejects_zero_masked():
    lines = _train_lines()
    idx = next(i for i, ln in enumerate(lines)
               if ln.strip() == "if roi_obs_telemetry:" and _indent(ln) == 8)
    # walk the guarded block and find the coverage condition
    body = []
    for ln in lines[idx + 1:]:
        if ln.strip() and _indent(ln) <= 8:
            break
        body.append(ln)
    assert any(ln.strip() == "if n_masked <= 0 or n_cm < n_masked:" for ln in body)
    assert any("[ROI-OBS]" in ln for ln in body)
    # replica truth table: zero masked views is now a hard failure
    def ok(n_cm, n_masked):
        return not (n_masked <= 0 or n_cm < n_masked)
    assert ok(4, 4) is True
    assert ok(5, 4) is True                 # more maps than masked views is fine
    assert ok(3, 4) is False
    assert ok(0, 0) is False                # NEW: 0/0 no longer passes vacuously
    assert ok(5, 0) is False


def test_train_py_dump_is_written_atomically():
    lines = _train_lines()
    src = "\n".join(lines)
    save = next(i for i, ln in enumerate(lines) if "np.savez_compressed(" in ln)
    repl = next(i for i, ln in enumerate(lines) if "os.replace(tmp_path, npz_path)" in ln)
    assert "np.savez_compressed(tmp_path, **arrs)" in lines[save]
    assert save < repl                       # write the temp file, then swap it in
    assert "np.savez_compressed(npz_path" not in src
    assert 'tmp_path = os.path.join(dataset.model_path, "obs_telemetry_tmp.npz")' in src
    assert 'npz_path = os.path.join(dataset.model_path, "obs_telemetry.npz")' in src
    # numpy only appends ".npz" when missing; the temp name already ends in .npz
    assert lines[repl].strip().startswith("os.replace(")


def test_train_py_flag_requires_use_roi_masks():
    src = "\n".join(_train_lines())
    assert re.search(r"if roi_obs_telemetry and not roi_enabled:\s*\n\s*raise RuntimeError",
                     src)


# ---------------------------------------------------------------------------
# loadCam want_class_map condition (now also true for --roi_obs_telemetry)
# ---------------------------------------------------------------------------

def _want_class_map_expr():
    """Pull loadCam's want_class_map expression out of the source, balanced-paren."""
    src = inspect.getsource(camera_utils.loadCam)
    marker = "want_class_map = ("
    idx = src.index(marker) + len(marker) - 1          # position of the '('
    depth = 0
    for j in range(idx, len(src)):
        if src[j] == "(":
            depth += 1
        elif src[j] == ")":
            depth -= 1
            if depth == 0:
                return src[idx:j + 1]
    raise AssertionError("unbalanced parens in loadCam want_class_map expression")


@pytest.mark.parametrize("cw,sr,obs,expected", [
    ("", "", False, False),             # all off -> byte-identical old path
    (None, None, False, False),
    ("", "", True, True),               # NEW: telemetry alone must request the map
    (None, None, True, True),
    ("2:2.0", "", False, True),         # densify weighting alone (pre-existing)
    ("", "2:0.01", False, True),        # scale reg alone (pre-existing)
    ("2:2.0", "2:0.01", True, True),
])
def test_loadcam_want_class_map_condition(cw, sr, obs, expected):
    expr = _want_class_map_expr()
    assert "roi_obs_telemetry" in expr
    assert "roi_scale_reg" in expr and "roi_densify_class_weights" in expr
    args = SimpleNamespace(roi_densify_class_weights=cw, roi_scale_reg=sr,
                           roi_obs_telemetry=obs)
    assert eval(expr, {"args": args}) is expected


def test_loadcam_want_class_map_defaults_off():
    # Missing attributes entirely (a namespace built before any of the flags existed).
    assert eval(_want_class_map_expr(), {"args": SimpleNamespace()}) is False


def test_roi_obs_telemetry_flag_defaults_false():
    from arguments import ModelParams
    src = inspect.getsource(ModelParams.__init__)
    assert "self.roi_obs_telemetry = False" in src
