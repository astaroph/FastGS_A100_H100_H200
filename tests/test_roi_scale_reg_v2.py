"""Tests for the r4 scale regularization v2 (FASTGS_ROI_SCALE_REG_V2).

Covers the new pieces plus the train.py / arguments plumbing that wires them up:
  * fast_utils.scale_reg_v2_penalty      -- one-sided term1 (hiddenness hinge) /
                                            term2 (plate floor), eligibility gating
  * fast_utils.observability_telemetry(with_billboard=False) -- kNN-skip fast path
  * loadCam's want_class_map condition (now also true for --roi_scale_reg_v2)
  * the train.py [ROI-SCALE-REG-V2] startup/loss-site/refresh guards
  * arguments/__init__.py's roi_scale_reg_v2* dials (ModelParams / OptimizationParams)
  * roi_utils.parse_scale_reg_spec's single-entry shape that the v2 label-only
    contract relies on

Everything is CPU-only: render_fastgs is monkeypatched, and the rotation build used
inside observability_telemetry is a CPU mirror of utils.general_utils.build_rotation
(which hardcodes device='cuda').
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
from utils import camera_utils, fast_utils, general_utils, roi_utils

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# build_rotation: CPU mirror (the shipped one allocates on 'cuda'); needed for
# observability_telemetry, which scale_reg_v2's with_billboard tests exercise.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# scale_reg_v2_penalty: signature contract
# ---------------------------------------------------------------------------

LOG_R0 = math.log(4.0)
LOG_RP0 = math.log(3.0)


def test_scale_reg_v2_penalty_signature_defaults():
    sig = inspect.signature(fast_utils.scale_reg_v2_penalty)
    assert sig.parameters["eps"].default == 1e-3
    assert sig.parameters["with_stats"].default is False


# ---------------------------------------------------------------------------
# scale_reg_v2_penalty: term1 math (fires / silent / h-modulated threshold)
# ---------------------------------------------------------------------------

def test_term1_math_fires_silent_and_h_modulated_threshold():
    # 3 gaussians, unsorted scale columns (exercises the sort permutation):
    #   g0: h~1 -> allow ~= LOG_R0; ls1-ls2 = log10-log2, well above -> fires
    #   g1: h~1 same allow; ls1-ls2 = log3-0, well below -> silent
    #   g2: h=9.99 -> allow is NEGATIVE (LOG_R0 - log(~10)); tied top-two scales
    #       (diff 0) still fire because the allowance itself is negative
    # Mean is over all 3 eligible rows, including g1's exact zero.
    cols = torch.tensor([
        [math.log(2.0), math.log(10.0), -5.0],
        [0.0, math.log(3.0), -1.0],
        [5.0, 5.0, -2.0],
    ], dtype=torch.float32)
    h_vals = [0.999, 0.999, 9.99]
    h = torch.tensor(h_vals, dtype=torch.float32)
    elig1 = torch.tensor([True, True, True])
    elig2 = torch.zeros(3, dtype=torch.bool)
    eps = 1e-3

    def expected_pen(ls1, ls2, hv):
        allow = LOG_R0 - math.log(hv + eps)
        return max(0.0, ls1 - ls2 - allow)

    expected = [
        expected_pen(math.log(10.0), math.log(2.0), h_vals[0]),
        expected_pen(math.log(3.0), 0.0, h_vals[1]),
        expected_pen(5.0, 5.0, h_vals[2]),
    ]
    assert expected[1] == 0.0                       # sanity: truly silent, not tiny
    assert expected[2] > 0.0                        # sanity: h alone opened this hinge

    out = fast_utils.scale_reg_v2_penalty(cols, h, elig1, elig2, LOG_R0, LOG_RP0,
                                          with_stats=True)
    assert float(out["term1"]) == pytest.approx(sum(expected) / 3.0, abs=1e-5)
    assert out["n1"] == 3
    assert out["act1"] == pytest.approx(2.0 / 3.0, abs=1e-6)   # g0, g2 fire; g1 doesn't
    assert float(out["term2"]) == 0.0 and out["n2"] == 0 and out["act2"] == 0.0


# ---------------------------------------------------------------------------
# scale_reg_v2_penalty: term1 one-sidedness via autograd
# ---------------------------------------------------------------------------

def test_term1_gradient_flows_only_to_ls1_argmax_column():
    # Unsorted columns exercise the sort permutation. Rows cover: an active
    # elig1 row, an eligible-but-inactive elig1 row, an elig2-only row (must
    # NEVER leak into term1 even though it would fire if it did), and a fully
    # ineligible row (excluded before any math at all).
    scaling = torch.tensor([
        [0.5, 3.0, -2.0],      # row0: elig1 active  -> ls1=col1(3.0) ls2=col0(0.5)
        [0.2, 0.1, -0.3],      # row1: elig1 inactive -> ls1=col0(0.2) ls2=col1(0.1)
        [5.0, 50.0, -5.0],     # row2: elig2-only, would fire hugely if wrongly counted
        [9.0, 9.0, 9.0],       # row3: fully ineligible
    ], dtype=torch.float32, requires_grad=True)
    h = torch.full((4,), 0.999, dtype=torch.float32)
    elig1 = torch.tensor([True, True, False, False])
    elig2 = torch.tensor([False, False, True, False])

    out = fast_utils.scale_reg_v2_penalty(scaling, h, elig1, elig2, LOG_R0, LOG_RP0)
    out["term1"].backward()

    expected_grad = torch.tensor([
        [0.0, 0.5, 0.0],       # 1/n1 (n1=2) landed on the argmax column only
        [0.0, 0.0, 0.0],       # inactive: relu closed, zero everywhere
        [0.0, 0.0, 0.0],       # elig2-only: masked out of term1 entirely
        [0.0, 0.0, 0.0],       # not in elig1|elig2 at all
    ])
    assert torch.allclose(scaling.grad, expected_grad, atol=1e-6)


# ---------------------------------------------------------------------------
# scale_reg_v2_penalty: term2 math + one-sidedness via autograd
# ---------------------------------------------------------------------------

def test_term2_math_hinge_and_grad_pushes_ls3_up_only():
    # Rows cover: an active elig2 row, an eligible-but-inactive elig2 row, an
    # elig1-only row (must never leak into term2), and a fully ineligible row.
    scaling = torch.tensor([
        [math.log(5.0), math.log(30.0), 0.0],             # ls1=log30 ls2=log5 ls3=0
        [math.log(5.0), math.log(6.0), math.log(4.0)],    # ls1=log6  ls2=log5 ls3=log4
        [math.log(100.0), math.log(50.0), 0.0],           # elig1-only, huge diff
        [1.0, 1.0, 1.0],                                   # fully ineligible
    ], dtype=torch.float32, requires_grad=True)
    h = torch.full((4,), 0.999, dtype=torch.float32)
    elig1 = torch.tensor([False, False, True, False])
    elig2 = torch.tensor([True, True, False, False])

    out = fast_utils.scale_reg_v2_penalty(scaling, h, elig1, elig2, LOG_R0, LOG_RP0)
    expected_term2 = (math.log(5.0) - LOG_RP0 + 0.0) / 2.0   # row0 fires, row1 silent
    assert float(out["term2"].detach()) == pytest.approx(expected_term2, abs=1e-6)

    out["term2"].backward()
    expected_grad = torch.tensor([
        [0.0, 0.0, -0.5],      # -1/n2 (n2=2) landed on the min-scale column only
        [0.0, 0.0, 0.0],       # inactive: relu closed
        [0.0, 0.0, 0.0],       # elig1-only: masked out of term2 entirely
        [0.0, 0.0, 0.0],       # not in elig1|elig2 at all
    ])
    assert torch.allclose(scaling.grad, expected_grad, atol=1e-6)
    # sign check restated explicitly: a negative gradient means gradient-descent
    # INCREASES ls3 (grows the thin axis toward the mid axis / plate_ratio).
    assert float(scaling.grad[0, 2]) < 0.0


# ---------------------------------------------------------------------------
# scale_reg_v2_penalty: elig1 / elig2 never cross-contaminate
# ---------------------------------------------------------------------------

def test_elig1_and_elig2_rows_never_cross_contaminate():
    # Row X: elig1 only, term1 fires. Row Y: elig2 only, term2 fires. Each
    # term's mean must equal exactly that one row's contribution -- proof the
    # other row (eligible for the OTHER term only) never leaks in.
    scaling = torch.tensor([
        [math.log(10.0), math.log(2.0), 0.0],
        [math.log(5.0), math.log(30.0), 0.0],
    ], dtype=torch.float32)
    h = torch.full((2,), 0.999, dtype=torch.float32)
    elig1 = torch.tensor([True, False])
    elig2 = torch.tensor([False, True])

    out = fast_utils.scale_reg_v2_penalty(scaling, h, elig1, elig2, LOG_R0, LOG_RP0,
                                          with_stats=True)
    assert out["n1"] == 1 and out["n2"] == 1
    assert float(out["term1"]) == pytest.approx(
        math.log(10.0) - math.log(2.0) - LOG_R0, abs=1e-6)
    assert float(out["term2"]) == pytest.approx(
        math.log(5.0) - LOG_RP0, abs=1e-6)


# ---------------------------------------------------------------------------
# scale_reg_v2_penalty: empty eligibility / eps keeps log finite
# ---------------------------------------------------------------------------

def test_empty_eligibility_gives_zero_scalars_no_crash():
    scaling = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    h = torch.tensor([0.5, 0.7], dtype=torch.float32)
    elig1 = torch.zeros(2, dtype=torch.bool)
    elig2 = torch.zeros(2, dtype=torch.bool)

    out = fast_utils.scale_reg_v2_penalty(scaling, h, elig1, elig2, LOG_R0, LOG_RP0,
                                          with_stats=True)
    assert torch.equal(out["term1"], torch.zeros(()))
    assert torch.equal(out["term2"], torch.zeros(()))
    assert out["term1"].shape == () and out["term2"].shape == ()
    assert out["act1"] == 0.0 and out["act2"] == 0.0
    assert out["n1"] == 0 and out["n2"] == 0


def test_h_zero_keeps_log_finite_via_eps():
    # h=0.0 exactly: log(h+eps) = log(1e-3), finite (not -inf). The resulting
    # huge negative allowance means a modest scale gap stays silent, matching
    # the docstring's "h -> 0 makes the allowance huge" self-gating claim.
    scaling = torch.tensor([[math.log(3.0), 0.0, -1.0]], dtype=torch.float32)
    h = torch.tensor([0.0], dtype=torch.float32)
    elig1 = torch.tensor([True])
    elig2 = torch.tensor([False])

    out = fast_utils.scale_reg_v2_penalty(scaling, h, elig1, elig2, LOG_R0, LOG_RP0)
    assert math.isfinite(float(out["term1"]))
    assert float(out["term1"]) == 0.0


# ---------------------------------------------------------------------------
# scale_reg_v2_penalty: with_stats gating
# ---------------------------------------------------------------------------

def test_with_stats_true_fills_act_and_n_false_leaves_defaults():
    scaling = torch.tensor([
        [math.log(10.0), math.log(2.0), 0.0],    # active term1
        [math.log(3.0), 0.0, -1.0],              # inactive term1
        [math.log(5.0), math.log(30.0), 0.0],    # active term2
    ], dtype=torch.float32)
    h = torch.full((3,), 0.999, dtype=torch.float32)
    elig1 = torch.tensor([True, True, False])
    elig2 = torch.tensor([False, False, True])

    off = fast_utils.scale_reg_v2_penalty(scaling, h, elig1, elig2, LOG_R0, LOG_RP0,
                                          with_stats=False)
    assert off["act1"] == 0.0 and off["act2"] == 0.0
    assert off["n1"] == 0 and off["n2"] == 0

    on = fast_utils.scale_reg_v2_penalty(scaling, h, elig1, elig2, LOG_R0, LOG_RP0,
                                         with_stats=True)
    assert on["n1"] == 2 and on["n2"] == 1
    assert on["act1"] == pytest.approx(0.5, abs=1e-9)    # 1 of 2 elig1 rows fires
    assert on["act2"] == pytest.approx(1.0, abs=1e-9)    # the only elig2 row fires


# ---------------------------------------------------------------------------
# scale_reg_v2_penalty: NaN-safety of fully-ineligible rows
# ---------------------------------------------------------------------------

def test_ineligible_rows_may_hold_garbage_without_affecting_results():
    # Rows outside elig1|elig2 are excluded by the row-subset (idx = nonzero
    # (elig1|elig2)) before any math touches them, so garbage there (NaN h,
    # +-inf/huge scales) must not change the result and must not crash.
    base_scaling = torch.tensor([
        [math.log(10.0), math.log(2.0), 0.0],
        [math.log(5.0), math.log(30.0), 0.0],
    ], dtype=torch.float32)
    base_h = torch.tensor([0.999, 0.999], dtype=torch.float32)
    base_elig1 = torch.tensor([True, False])
    base_elig2 = torch.tensor([False, True])
    base = fast_utils.scale_reg_v2_penalty(base_scaling, base_h, base_elig1, base_elig2,
                                           LOG_R0, LOG_RP0, with_stats=True)

    garbage_rows = torch.tensor([
        [float("inf"), float("-inf"), 1e30],
        [-1e30, float("nan"), 5.0],
    ], dtype=torch.float32)
    aug_scaling = torch.cat([base_scaling, garbage_rows], dim=0)
    aug_h = torch.cat([base_h, torch.tensor([float("nan"), 1e30], dtype=torch.float32)])
    aug_elig1 = torch.cat([base_elig1, torch.tensor([False, False])])
    aug_elig2 = torch.cat([base_elig2, torch.tensor([False, False])])

    aug = fast_utils.scale_reg_v2_penalty(aug_scaling, aug_h, aug_elig1, aug_elig2,
                                          LOG_R0, LOG_RP0, with_stats=True)

    assert torch.equal(base["term1"], aug["term1"])
    assert torch.equal(base["term2"], aug["term2"])
    assert base["n1"] == aug["n1"] and base["n2"] == aug["n2"]
    assert base["act1"] == aug["act1"] and base["act2"] == aug["act2"]
    assert math.isfinite(float(aug["term1"])) and math.isfinite(float(aug["term2"]))


# ---------------------------------------------------------------------------
# observability_telemetry(with_billboard=False): fixtures (minimal replica of
# the sibling test_roi_obs_telemetry.py fake-render table pattern)
# ---------------------------------------------------------------------------

ARGS = SimpleNamespace(mult=0.5)

LAB_ROWS = (2, 3)              # class 2 occupies rows 2-3 of _cm()
REM_ROWS = (0, 1, 4, 5)        # everything else


def _cm():
    # 6x6: rows 0-1 class 1 (mount), rows 2-3 class 2 (label), row 4 class 3
    # (specimen), row 5 class 0 (background). Same helper as the sibling files.
    m = torch.zeros((6, 6), dtype=torch.uint8)
    m[0:2, :] = 1
    m[2:4, :] = 2
    m[4, :] = 3
    return m


def _rows_key(metric_map):
    """Row indices touched by a stencil -- a stable, readable stencil identity."""
    sel = metric_map > 0
    return tuple(int(r) for r in torch.nonzero(sel.any(dim=1)).flatten().tolist())


def _fake_render(calls, table):
    """render_fastgs stand-in whose per-gaussian counts are chosen per stencil.

    table maps (image_name, touched_rows) -> per-gaussian counts. An unexpected
    render raises KeyError, so the tests also pin down exactly WHICH stencils get
    rendered.
    """
    def fake(cam, gaussians, pipe, bg, mult, get_flag=None, metric_map=None):
        assert get_flag is True
        assert metric_map.dtype == torch.int32
        assert mult == ARGS.mult
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
SCALE_XYZ = [1.0, 0.5, 0.0]


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


def _grid(nx, ny, z=0.0, dx=1.0):
    """nx*ny points on a z=const plane, spacing dx (row-major in x)."""
    pts = [[i * dx, j * dx, z] for i in range(nx) for j in range(ny)]
    return torch.tensor(pts, dtype=torch.float32)


def _flat_pool_scenario(n_pool):
    """n_pool coplanar, fully high-purity/single-camera-supported gaussians --
    a pool this size (> K_LOCAL=16) is exactly where with_billboard=True would
    otherwise run the kNN pass."""
    xyz = _grid(5, 5)[:n_pool]
    gaussians = _gauss(xyz)
    table = {("A", LAB_ROWS): [20] * n_pool, ("A", REM_ROWS): [0] * n_pool}
    return gaussians, table


# ---------------------------------------------------------------------------
# observability_telemetry(with_billboard=False)
# ---------------------------------------------------------------------------

def test_with_billboard_false_never_calls_local_plane_normals(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("local_plane_normals must not be called")

    monkeypatch.setattr(fast_utils, "local_plane_normals", boom)
    calls = []
    gaussians, table = _flat_pool_scenario(25)      # pool 25 > K_LOCAL (16)
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_render(calls, table))

    tel = fast_utils.observability_telemetry(
        [_cam("A", (0.0, 0.0, 10.0))], gaussians, None, None, ARGS, 2,
        with_billboard=False)

    assert tel["pool_mask"].all()                   # would have triggered the kNN pass
    assert torch.isnan(tel["billboard"]).all()


def test_with_billboard_false_matches_true_on_every_other_field(monkeypatch):
    calls = []
    gaussians, table = _flat_pool_scenario(25)
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_render(calls, table))

    tel_true = fast_utils.observability_telemetry(
        [_cam("A", (0.0, 0.0, 10.0))], gaussians, None, None, ARGS, 2,
        with_billboard=True)
    tel_false = fast_utils.observability_telemetry(
        [_cam("A", (0.0, 0.0, 10.0))], gaussians, None, None, ARGS, 2,
        with_billboard=False)

    assert torch.isfinite(tel_true["billboard"]).all()   # sanity: the kNN pass DID run
    assert torch.isnan(tel_false["billboard"]).all()
    for key in ("counts", "purity", "support", "rho", "c1", "c2", "h", "pool_mask"):
        assert torch.equal(tel_true[key], tel_false[key]), key
    assert tel_true["margin"] is None and tel_false["margin"] is None


# ---------------------------------------------------------------------------
# loadCam want_class_map condition (now also true for --roi_scale_reg_v2)
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


@pytest.mark.parametrize("cw,sr,v2,obs,expected", [
    ("", "", "", False, False),                 # all off -> byte-identical old path
    (None, None, None, False, False),
    ("", "", "2:0.003", False, True),           # NEW: v2 alone must request the map
    (None, None, "2:0.003", False, True),
    ("2:2.0", "", "", False, True),             # densify weighting alone (pre-existing)
    ("", "2:0.01", "", False, True),            # v1 scale reg alone (pre-existing)
    ("", "", "", True, True),                   # telemetry alone (pre-existing)
    ("2:2.0", "2:0.01", "2:0.003", True, True),
])
def test_loadcam_want_class_map_condition_with_v2(cw, sr, v2, obs, expected):
    expr = _want_class_map_expr()
    assert "roi_scale_reg_v2" in expr
    assert "roi_scale_reg" in expr and "roi_densify_class_weights" in expr
    assert "roi_obs_telemetry" in expr
    args = SimpleNamespace(roi_densify_class_weights=cw, roi_scale_reg=sr,
                           roi_scale_reg_v2=v2, roi_obs_telemetry=obs)
    assert eval(expr, {"args": args}) is expected


def test_loadcam_want_class_map_defaults_off():
    # Missing attributes entirely (a namespace built before any of the flags existed).
    assert eval(_want_class_map_expr(), {"args": SimpleNamespace()}) is False


# ---------------------------------------------------------------------------
# train.py source-text assertions
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


def test_v2_spec_is_parsed_inside_the_startup_on_guard():
    lines = _train_lines()
    guard_idx = next(i for i, ln in enumerate(lines) if ln.strip() == "if scale_reg_v2_on:")
    call_idx = next(i for i, ln in enumerate(lines)
                    if "parse_scale_reg_spec(scale_reg_v2_spec)" in ln)
    headers = _enclosing_headers(lines, call_idx)
    assert "if scale_reg_v2_on:" in headers
    assert call_idx > guard_idx
    assert _indent(lines[call_idx]) > _indent(lines[guard_idx])


def test_v2_loss_site_call_is_guarded_and_precedes_backward():
    lines = _train_lines()
    call_idx = next(i for i, ln in enumerate(lines) if "scale_reg_v2_penalty(" in ln)
    headers = _enclosing_headers(lines, call_idx)
    assert "if scale_reg_v2_on and scale_reg_v2_h is not None:" in headers
    backward_idx = next(i for i, ln in enumerate(lines) if ln.strip() == "loss.backward()")
    assert call_idx < backward_idx


def test_v2_observability_refresh_has_exactly_two_guarded_call_sites():
    # Two refresh sites: the densify-block refresh and the final-prune refresh.
    # Both must be guarded directly by "if scale_reg_v2_on:" (not the unrelated
    # "if roi_obs_telemetry:" heartbeat/dump call sites) and both must pass the
    # plate-term with_billboard switch.
    lines = _train_lines()
    obs_idxs = [i for i, ln in enumerate(lines) if "observability_telemetry(" in ln]
    guarded = [i for i in obs_idxs
              if _enclosing_headers(lines, i)[:1] == ["if scale_reg_v2_on:"]]
    assert len(guarded) == 2, guarded
    for i in guarded:
        window = "\n".join(lines[i:i + 6])
        assert "with_billboard=(scale_reg_v2_plate_lam > 0.0)" in window


def test_v2_mutual_exclusion_raise_is_inside_startup_guard():
    lines = _train_lines()
    guard_idx = next(i for i, ln in enumerate(lines) if ln.strip() == "if scale_reg_v2_on:")
    guard_indent = _indent(lines[guard_idx])
    body = []
    for ln in lines[guard_idx + 1:]:
        if ln.strip() and _indent(ln) <= guard_indent:
            break
        body.append(ln)
    src = "\n".join(body)
    assert re.search(r"mutually[^a-zA-Z]*exclusive", src)


# ---------------------------------------------------------------------------
# arguments/__init__.py: roi_scale_reg_v2* dials
# ---------------------------------------------------------------------------

def test_model_params_declares_roi_scale_reg_v2_before_super_init():
    from arguments import ModelParams
    src = inspect.getsource(ModelParams.__init__)
    decl_idx = src.index('self.roi_scale_reg_v2 = ""')
    super_idx = src.index('super().__init__(parser, "Loading Parameters"')
    assert decl_idx < super_idx


def test_optimization_params_v2_dials_have_frozen_defaults():
    from arguments import OptimizationParams
    src = inspect.getsource(OptimizationParams.__init__)
    assert "self.roi_scale_reg_v2_ratio = 4.0" in src
    assert "self.roi_scale_reg_v2_plate_lambda = 0.0" in src
    assert "self.roi_scale_reg_v2_plate_ratio = 150.0" in src


# ---------------------------------------------------------------------------
# parse_scale_reg_spec: the exact single-entry shape the v2 label-only
# contract relies on (v1 grouping/rejection behavior is covered by
# test_roi_scale_reg.py; this just pins the "2:<lambda>" shape v2 requires).
# ---------------------------------------------------------------------------

def test_parse_scale_reg_spec_v2_single_entry_shape():
    assert roi_utils.parse_scale_reg_spec("2:0.003") == [(0.003, [2])]
