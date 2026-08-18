"""Tests for class-scoped ray-modulated scale regularization (FASTGS_ROI_SCALE_REG).

Covers the three new pieces plus the train.py penalty formula they feed:
  * roi_utils.parse_scale_reg_spec           -- spec grouping / rejection rules
  * fast_utils._stencil_counts               -- binary get_flag render passthrough
  * fast_utils.attribute_gaussians_by_class  -- argmax attribution + direction resultant
  * the train.py [ROI-SCALE-REG] loss-site penalty, replicated in pure torch
  * loadCam's want_class_map condition (now also true for --roi_scale_reg)

Everything is CPU-only: render_fastgs is monkeypatched, and the rotation build is a
CPU mirror of utils.general_utils.build_rotation (which hardcodes device='cuda');
a CUDA-gated test asserts the mirror is numerically identical to the real thing.
"""
import inspect
import math
import os
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

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
from utils import camera_utils, fast_utils, roi_utils

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# parse_scale_reg_spec
# ---------------------------------------------------------------------------

def test_parse_groups_by_lambda_first_appearance_order():
    assert roi_utils.parse_scale_reg_spec("2:0.01,0:0.01") == [(0.01, [2, 0])]


def test_parse_distinct_lambdas_stay_separate_in_order():
    assert roi_utils.parse_scale_reg_spec("2:0.02,0:0.01") == [(0.02, [2]), (0.01, [0])]


def test_parse_keeps_lambda_one():
    # Unlike parse_densify_class_weights, 1.0 is a real lambda here, not identity.
    assert roi_utils.parse_scale_reg_spec("2:1.0") == [(1.0, [2])]


def test_parse_drops_zero_lambda_entries():
    assert roi_utils.parse_scale_reg_spec("2:0.01,4:0.0") == [(0.01, [2])]


@pytest.mark.parametrize("bad", [
    "",                 # empty
    None,               # None
    "2",                # no colon
    "2:1:0",            # extra colon
    "x:1.0",            # non-int id
    "2:abc",            # non-float lambda
    "300:0.01",         # id out of range
    "2:0.01,2:0.02",    # duplicate id
    "2:-0.5",           # negative lambda
    "2:inf",            # non-finite lambda
    "2:0.0",            # single all-zero entry -> fully disabled
    "2:0.0,3:0.0",      # every entry zero -> fully disabled
])
def test_parse_rejects(bad):
    with pytest.raises(ValueError):
        roi_utils.parse_scale_reg_spec(bad)


# ---------------------------------------------------------------------------
# _stencil_counts
# ---------------------------------------------------------------------------

def test_stencil_counts_binary_render_passthrough(monkeypatch):
    captured = {}

    def fake(cam, gaussians, pipe, bg, mult, get_flag=None, metric_map=None):
        assert get_flag is True
        # The CUDA kernel compares metric_map[pix] == 1 exactly; .int() must be int32.
        assert metric_map.dtype == torch.int32
        n = gaussians.get_xyz.shape[0]
        out = torch.full((n,), int(metric_map.sum().item()), dtype=torch.int32)
        captured["out"] = out
        captured["mult"] = mult
        return {"accum_metric_counts": out}

    monkeypatch.setattr(fast_utils, "render_fastgs", fake)
    stencil = torch.zeros((4, 4), dtype=torch.bool)
    stencil[1, :] = True
    stencil[2, 0:3] = True                       # 4 + 3 = 7 selected pixels
    gaussians = SimpleNamespace(get_xyz=torch.zeros((5, 3)))
    out = fast_utils._stencil_counts(
        SimpleNamespace(image_name="t"), gaussians, None, None,
        SimpleNamespace(mult=0.5), stencil)
    assert out is captured["out"]                # exact passthrough, no copy/cast
    assert captured["mult"] == 0.5
    assert out.dtype == torch.int32 and out.shape == (5,)
    assert torch.equal(out, torch.full((5,), 7, dtype=torch.int32))


def test_stencil_counts_casts_uint8_stencil_to_int32(monkeypatch):
    seen = {}

    def fake(cam, gaussians, pipe, bg, mult, get_flag=None, metric_map=None):
        assert get_flag is True
        seen["dtype"] = metric_map.dtype
        seen["sum"] = int(metric_map.sum().item())
        return {"accum_metric_counts": torch.zeros(2, dtype=torch.int32)}

    monkeypatch.setattr(fast_utils, "render_fastgs", fake)
    stencil = torch.zeros((3, 3), dtype=torch.uint8)
    stencil[0, 0:2] = 1
    fast_utils._stencil_counts(
        SimpleNamespace(image_name="t"), SimpleNamespace(get_xyz=torch.zeros((2, 3))),
        None, None, SimpleNamespace(mult=1.0), stencil)
    assert seen == {"dtype": torch.int32, "sum": 2}


# ---------------------------------------------------------------------------
# attribute_gaussians_by_class (render_fastgs monkeypatched)
# ---------------------------------------------------------------------------

def _cm():
    # 6x6: rows 0-1 class 1 (mount), rows 2-3 class 2 (label), row 4 class 3
    # (specimen), row 5 class 0 (background). Same helper as the sibling file.
    m = torch.zeros((6, 6), dtype=torch.uint8)
    m[0:2, :] = 1
    m[2:4, :] = 2
    m[4, :] = 3
    return m


def _rows_key(metric_map):
    """Row indices touched by a stencil -- a stable, readable stencil identity."""
    sel = metric_map > 0
    return tuple(int(r) for r in torch.nonzero(sel.any(dim=1)).flatten().tolist())


def _fake_attr_render(calls, table):
    """render_fastgs stand-in whose per-gaussian counts are chosen per stencil.

    table maps (image_name, touched_rows) -> list of per-gaussian counts. An
    unexpected render raises KeyError, so the tests also pin down exactly WHICH
    stencils get rendered.
    """
    def fake(cam, gaussians, pipe, bg, mult, get_flag=None, metric_map=None):
        assert get_flag is True
        assert metric_map.dtype == torch.int32
        key = (cam.image_name, _rows_key(metric_map))
        calls.append(key)
        return {"accum_metric_counts": torch.tensor(table[key], dtype=torch.int32)}
    return fake


def _cam(name, class_map, center):
    return SimpleNamespace(roi_class_map=class_map, image_name=name,
                           camera_center=torch.tensor(center, dtype=torch.float32))


ARGS = SimpleNamespace(mult=0.5)


def test_attribution_argmax_group_remainder_tie_and_zero(monkeypatch):
    # groups = [(0.01, [2])] -> col0 = class-2 group, col1 = remainder.
    # class 2 occupies rows 2-3; the remainder stencil is rows 0,1,4,5.
    calls = []
    table = {
        ("A", (2, 3)): [10, 3, 4, 0],           # group counts
        ("A", (0, 1, 4, 5)): [5, 7, 4, 0],      # remainder counts
    }
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_attr_render(calls, table))
    gaussians = SimpleNamespace(get_xyz=torch.zeros((4, 3)))
    attr, r_bar = fast_utils.attribute_gaussians_by_class(
        [_cam("A", _cm(), (0.0, 0.0, 1.0))], gaussians, None, None, ARGS,
        [(0.01, [2])])
    # g0: 10 > 5 -> group 0 ; g1: remainder wins -> -1 ; g2: 4 == 4 tie -> -1 ;
    # g3: never rendered onto (0/0) -> -1
    assert attr.tolist() == [0, -1, -1, -1]
    # single camera at (0,0,1), all gaussians at the origin -> unit dir (0,0,1);
    # r_bar = cam_counts * dir / cam_counts = dir, except the all-zero row (0/1).
    expected = torch.tensor([[0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 0.]])
    assert torch.allclose(r_bar, expected)
    assert sorted(calls) == sorted([("A", (2, 3)), ("A", (0, 1, 4, 5))])
    assert len(calls) == 2                       # one per group + one remainder


def test_attribution_returns_declared_dtypes_and_shapes(monkeypatch):
    calls = []
    table = {("A", (2, 3)): [1, 1], ("A", (0, 1, 4, 5)): [0, 0]}
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_attr_render(calls, table))
    attr, r_bar = fast_utils.attribute_gaussians_by_class(
        [_cam("A", _cm(), (1.0, 0.0, 0.0))], SimpleNamespace(get_xyz=torch.zeros((2, 3))),
        None, None, ARGS, [(0.01, [2])])
    assert attr.dtype == torch.int8 and attr.shape == (2,)
    assert r_bar.dtype == torch.float32 and r_bar.shape == (2, 3)


def test_rbar_opposite_cameras_equal_counts_cancel(monkeypatch):
    # Two opposite cameras with EQUAL total counts -> resultant ~ 0 (rho ~ 0).
    calls = []
    table = {
        ("A", (2, 3)): [4], ("A", (0, 1, 4, 5)): [1],     # cam A total 5
        ("B", (2, 3)): [2], ("B", (0, 1, 4, 5)): [3],     # cam B total 5
    }
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_attr_render(calls, table))
    cams = [_cam("A", _cm(), (2.0, 0.0, 0.0)), _cam("B", _cm(), (-3.0, 0.0, 0.0))]
    attr, r_bar = fast_utils.attribute_gaussians_by_class(
        cams, SimpleNamespace(get_xyz=torch.zeros((1, 3))), None, None, ARGS,
        [(0.5, [2])])
    # 5*(+1,0,0) + 5*(-1,0,0) = 0, divided by wsum 10
    assert torch.allclose(r_bar, torch.zeros((1, 3)), atol=1e-7)
    assert float(r_bar.norm(dim=1)) == pytest.approx(0.0, abs=1e-7)
    # 6 group px (class 2, col 0) vs 4 remainder px totals: 4+2=6 vs 1+3=4 -> group
    assert attr.tolist() == [0]
    assert len(calls) == 4


def test_rbar_single_camera_is_unit_length(monkeypatch):
    calls = []
    table = {("A", (2, 3)): [7], ("A", (0, 1, 4, 5)): [3]}
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_attr_render(calls, table))
    attr, r_bar = fast_utils.attribute_gaussians_by_class(
        [_cam("A", _cm(), (0.0, 3.0, 4.0))], SimpleNamespace(get_xyz=torch.zeros((1, 3))),
        None, None, ARGS, [(0.01, [2])])
    assert torch.allclose(r_bar, torch.tensor([[0.0, 0.6, 0.8]]), atol=1e-6)
    assert float(r_bar.norm(dim=1)) == pytest.approx(1.0, abs=1e-6)


def test_rbar_is_count_weighted_and_not_renormalized(monkeypatch):
    # cam A (+x) total 3, cam B (+y) total 1 -> r_bar = (0.75, 0.25, 0), rho < 1.
    calls = []
    table = {
        ("A", (2, 3)): [2], ("A", (0, 1, 4, 5)): [1],
        ("B", (2, 3)): [1], ("B", (0, 1, 4, 5)): [0],
    }
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_attr_render(calls, table))
    cams = [_cam("A", _cm(), (1.0, 0.0, 0.0)), _cam("B", _cm(), (0.0, 5.0, 0.0))]
    _attr, r_bar = fast_utils.attribute_gaussians_by_class(
        cams, SimpleNamespace(get_xyz=torch.zeros((1, 3))), None, None, ARGS,
        [(0.01, [2])])
    assert torch.allclose(r_bar, torch.tensor([[0.75, 0.25, 0.0]]), atol=1e-6)
    assert float(r_bar.norm(dim=1)) == pytest.approx(math.sqrt(0.75 ** 2 + 0.25 ** 2), abs=1e-6)


def test_failopen_camera_contributes_nothing(monkeypatch):
    # A camera whose roi_class_map is None must be skipped entirely: no renders,
    # no direction contribution (its center would otherwise pull r_bar to -x).
    calls = []
    table = {("A", (2, 3)): [7], ("A", (0, 1, 4, 5)): [3]}
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_attr_render(calls, table))
    cams = [_cam("A", _cm(), (0.0, 3.0, 4.0)), _cam("B", None, (-100.0, 0.0, 0.0))]
    attr, r_bar = fast_utils.attribute_gaussians_by_class(
        cams, SimpleNamespace(get_xyz=torch.zeros((1, 3))), None, None, ARGS,
        [(0.01, [2])])
    assert len(calls) == 2                                   # cam B rendered nothing
    assert all(name == "A" for name, _rows in calls)
    assert torch.allclose(r_bar, torch.tensor([[0.0, 0.6, 0.8]]), atol=1e-6)
    assert attr.tolist() == [0]                              # 7 > 3


def test_empty_stencil_group_triggers_no_render(monkeypatch):
    # Class 7 is absent from the map -> group 0 must render nothing (a KeyError in
    # the fake would fire if it did) and stays at zero counts.
    calls = []
    table = {("A", (2, 3)): [9, 1], ("A", (0, 1, 4, 5)): [2, 5]}
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_attr_render(calls, table))
    attr, _r_bar = fast_utils.attribute_gaussians_by_class(
        [_cam("A", _cm(), (0.0, 0.0, 1.0))], SimpleNamespace(get_xyz=torch.zeros((2, 3))),
        None, None, ARGS, [(0.01, [7]), (0.02, [2])])
    assert len(calls) == 2                                   # group 1 + remainder only
    assert ("A", (2, 3)) in calls and ("A", (0, 1, 4, 5)) in calls
    # g0: col1 (group of class 2) = 9 beats remainder 2 -> attr 1 ; g1: remainder wins
    assert attr.tolist() == [1, -1]


def test_multi_id_group_unions_stencils(monkeypatch):
    # groups = [(0.01, [2, 3])] -> one union stencil over rows 2-3 (class 2) and
    # row 4 (class 3); the remainder is rows 0,1,5.
    calls = []
    table = {("A", (2, 3, 4)): [6], ("A", (0, 1, 5)): [5]}
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_attr_render(calls, table))
    attr, r_bar = fast_utils.attribute_gaussians_by_class(
        [_cam("A", _cm(), (0.0, 0.0, 2.0))], SimpleNamespace(get_xyz=torch.zeros((1, 3))),
        None, None, ARGS, [(0.01, [2, 3])])
    assert len(calls) == 2
    assert attr.tolist() == [0]
    assert torch.allclose(r_bar, torch.tensor([[0.0, 0.0, 1.0]]), atol=1e-6)


# ---------------------------------------------------------------------------
# train.py [ROI-SCALE-REG] penalty math, replicated in pure torch
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


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="utils.general_utils.build_rotation hardcodes device='cuda'")
def test_cpu_rotation_mirror_matches_build_rotation():
    from utils.general_utils import build_rotation
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                          [0.5, 0.5, 0.5, 0.5],
                          [0.0, 0.70710678, 0.0, 0.70710678],
                          [2.0, -1.0, 0.5, 0.25]])
    ref = build_rotation(quats.cuda()).cpu()
    assert torch.allclose(_build_rotation_cpu(quats), ref, atol=1e-6)


def _scale_reg_penalty(scaling, rotation, r_bar, sub, log_r0):
    """Exact replica of the train.py loss-site block (lines around 391-405)."""
    ls_sub = scaling[sub]
    with torch.no_grad():
        rot_sub = _build_rotation_cpu(rotation[sub])
        amax = torch.argmax(ls_sub, dim=1)
        a1 = rot_sub[torch.arange(sub.numel(), device=sub.device), :, amax]
        align = (a1 * r_bar[sub]).sum(dim=1).abs().clamp_min(1e-3)
    top2 = torch.topk(ls_sub, 2, dim=1).values
    pen = torch.relu(top2[:, 0] - top2[:, 1] + torch.log(align) - log_r0)
    return pen, align, a1


IDENT_Q = torch.tensor([1.0, 0.0, 0.0, 0.0])
LOG_R0_4 = math.log(4.0)


def _one(ls, r_bar_vec, log_r0=LOG_R0_4):
    scaling = torch.tensor([ls], dtype=torch.float32)
    rotation = IDENT_Q[None, :].clone()
    r_bar = torch.tensor([r_bar_vec], dtype=torch.float32)
    return _scale_reg_penalty(scaling, rotation, r_bar, torch.tensor([0]), log_r0)


def test_identity_quaternion_a1_is_argmax_axis():
    # Identity rotation -> build_rotation column amax is the basis vector e_amax.
    _pen, _align, a1 = _one([0.0, 3.0, 1.0], [0.0, 1.0, 0.0])
    assert torch.allclose(a1, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)


def test_penalty_fires_above_threshold():
    # ls1 - ls2 = ln 10, align 1 (a1 == r_bar), r0 = 4 -> pen = ln(10/4) = ln 2.5
    pen, align, _a1 = _one([math.log(10.0), 0.0, 0.0], [1.0, 0.0, 0.0])
    assert float(align) == pytest.approx(1.0, abs=1e-6)
    assert float(pen) == pytest.approx(math.log(2.5), abs=1e-6)


def test_penalty_silent_below_threshold():
    # ln 3 < ln 4 -> hinge closed, exactly zero (not merely small).
    pen, _align, _a1 = _one([math.log(3.0), 0.0, 0.0], [1.0, 0.0, 0.0])
    assert float(pen) == 0.0


def test_penalty_at_threshold_is_zero():
    # ls1 - ls2 == ln r0 exactly -> relu(0) == 0 (hinge is strict).
    pen, _align, _a1 = _one([math.log(4.0), 0.0, 0.0], [1.0, 0.0, 0.0])
    assert float(pen) == pytest.approx(0.0, abs=1e-6)


def test_align_tenth_shifts_effective_threshold_by_ln10():
    # align 0.1 adds ln(0.1) = -ln 10 to the hinge, i.e. the tolerated scale ratio
    # grows by exactly 10x: ratio 100 at align 0.1 == ratio 10 at align 1.0.
    pen_a, align_a, _ = _one([math.log(10.0), 0.0, 0.0], [1.0, 0.0, 0.0])
    pen_c, align_c, _ = _one([math.log(100.0), 0.0, 0.0], [0.1, 0.0, 0.0])
    assert float(align_c) == pytest.approx(0.1, abs=1e-6)
    assert float(align_a) == pytest.approx(1.0, abs=1e-6)
    assert float(pen_c) == pytest.approx(float(pen_a), abs=1e-5)
    assert float(pen_c) == pytest.approx(math.log(2.5), abs=1e-5)
    # and the same ratio 10 that fired at align 1.0 is fully silenced at align 0.1
    pen_silent, _al, _a1 = _one([math.log(10.0), 0.0, 0.0], [0.1, 0.0, 0.0])
    assert float(pen_silent) == 0.0


def test_align_sign_is_absolute():
    # a1 . r_bar = -1 must behave identically to +1 (axis has no orientation).
    pen_pos, align_pos, _ = _one([math.log(10.0), 0.0, 0.0], [1.0, 0.0, 0.0])
    pen_neg, align_neg, _ = _one([math.log(10.0), 0.0, 0.0], [-1.0, 0.0, 0.0])
    assert float(align_neg) == pytest.approx(float(align_pos), abs=1e-6)
    assert float(pen_neg) == pytest.approx(float(pen_pos), abs=1e-6)


def test_clamp_floor_applies_when_rbar_orthogonal_to_a1():
    # r_bar perpendicular to the long axis -> dot 0 -> clamped to 1e-3, so the
    # penalty stays finite: 10.0 + ln(1e-3) - ln 4.
    pen, align, _a1 = _one([10.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    assert float(align) == pytest.approx(1e-3, abs=1e-9)
    expected = 10.0 + math.log(1e-3) - LOG_R0_4
    assert float(pen) == pytest.approx(expected, abs=1e-5)
    assert math.isfinite(float(pen)) and float(pen) > 0.0
    # a modest anisotropy is fully silenced by the same floor
    pen_small, _al, _a1b = _one([math.log(10.0), 0.0, 0.0], [0.0, 1.0, 0.0])
    assert float(pen_small) == 0.0


def test_equal_top_two_scales_give_zero_penalty():
    # s1 == s2 -> ls1 - ls2 == 0; align <= 1 and r0 > 1 so the hinge cannot open.
    pen, _align, _a1 = _one([2.0, 2.0, 1.0], [1.0, 0.0, 0.0])
    assert float(pen) == 0.0


def test_zero_rbar_row_cannot_open_hinge_for_isotropic_gaussian():
    # An unobserved row (r_bar all zero) clamps to 1e-3: pen 0 unless ls1-ls2 is
    # larger than ln r0 + ln 1000.
    pen, align, _a1 = _one([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert float(align) == pytest.approx(1e-3, abs=1e-9)
    assert float(pen) == 0.0


def test_gradient_reaches_only_top_two_scale_entries():
    # Row 0 (attributed): ls = [ln10, ln2, ln1] -> top2 = (ln10, ln2), fires.
    # Row 1: NOT attributed (excluded by sub) -> zero grad.
    # Row 2 (attributed): ls = [0, 3, 1] -> top2 = (3, 1) on entries 1 and 2.
    scaling = torch.tensor([[math.log(10.0), math.log(2.0), 0.0],
                            [5.0, 5.0, 5.0],
                            [0.0, 3.0, 1.0]], dtype=torch.float32, requires_grad=True)
    rotation = IDENT_Q[None, :].repeat(3, 1)
    r_bar = torch.tensor([[1.0, 0.0, 0.0],        # aligned with e0 (row 0 amax = 0)
                          [0.0, 0.0, 1.0],
                          [0.0, 1.0, 0.0]],       # aligned with e1 (row 2 amax = 1)
                         dtype=torch.float32)
    sub = torch.tensor([0, 2])
    pen, align, _a1 = _scale_reg_penalty(scaling, rotation, r_bar, sub, LOG_R0_4)
    assert torch.allclose(align, torch.tensor([1.0, 1.0]), atol=1e-6)
    pen_vals = pen.detach()
    assert float(pen_vals[0]) == pytest.approx(math.log(10.0 / 2.0) - LOG_R0_4, abs=1e-6)
    assert float(pen_vals[1]) == pytest.approx(3.0 - 1.0 - LOG_R0_4, abs=1e-6)
    pen.sum().backward()
    expected_grad = torch.tensor([[1.0, -1.0, 0.0],     # +1 on ls1, -1 on ls2, 0 on ls3
                                  [0.0, 0.0, 0.0],      # unattributed row untouched
                                  [0.0, 1.0, -1.0]])
    assert torch.allclose(scaling.grad, expected_grad, atol=1e-6)


def test_closed_hinge_produces_zero_gradient():
    scaling = torch.tensor([[math.log(3.0), 0.0, 0.0]], dtype=torch.float32,
                           requires_grad=True)
    pen, _align, _a1 = _scale_reg_penalty(
        scaling, IDENT_Q[None, :].clone(), torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([0]), LOG_R0_4)
    pen.sum().backward()
    assert torch.allclose(scaling.grad, torch.zeros((1, 3)))


def test_per_group_lambda_weighted_mean_loss_increment():
    # Replica of train.py's group loop: loss += lam_g * pen[attr == g].mean()
    scaling = torch.tensor([[math.log(10.0), math.log(2.0), 0.0],   # pen = ln(1.25)
                            [math.log(3.0), 0.0, 0.0],             # pen = 0
                            [math.log(8.0), 0.0, 0.0],             # pen = ln 2
                            [math.log(10.0), 0.0, 0.0]],           # pen = ln 2.5
                           dtype=torch.float32)
    rotation = IDENT_Q[None, :].repeat(4, 1)
    r_bar = torch.tensor([[1.0, 0.0, 0.0]] * 4, dtype=torch.float32)
    sub = torch.arange(4)
    pen, _align, _a1 = _scale_reg_penalty(scaling, rotation, r_bar, sub, LOG_R0_4)
    assert torch.allclose(
        pen, torch.tensor([math.log(1.25), 0.0, math.log(2.0), math.log(2.5)]), atol=1e-6)

    groups = [(0.5, [2]), (2.0, [3])]
    attr_sub = torch.tensor([0, 0, 1, 1], dtype=torch.int8)
    loss = torch.tensor(1.0)
    for gi, (lam, _ids) in enumerate(groups):
        gsel = attr_sub == gi
        if bool(gsel.any()):
            loss = loss + float(lam) * pen[gsel].mean()
    expected = (1.0
                + 0.5 * (math.log(1.25) + 0.0) / 2.0
                + 2.0 * (math.log(2.0) + math.log(2.5)) / 2.0)
    assert float(loss) == pytest.approx(expected, abs=1e-6)


def test_group_with_no_members_adds_nothing():
    pen = torch.tensor([0.5, 0.25])
    attr_sub = torch.tensor([0, 0], dtype=torch.int8)
    loss = torch.tensor(2.0)
    for gi, (lam, _ids) in enumerate([(0.5, [2]), (99.0, [3])]):   # group 1 empty
        gsel = attr_sub == gi
        if bool(gsel.any()):
            loss = loss + float(lam) * pen[gsel].mean()
    assert float(loss) == pytest.approx(2.0 + 0.5 * 0.375, abs=1e-7)


# ---------------------------------------------------------------------------
# loadCam / load_roi_products: class-map plumbing driven by --roi_scale_reg
# ---------------------------------------------------------------------------

LUT = roi_utils.parse_class_weights("0:0.0,1:0.3,2:1.0,3:1.0,4:1.0")


def _write_mask(tmp_path, arr):
    p = tmp_path / "mask.png"
    Image.fromarray(arr.numpy().astype(np.uint8), mode="L").save(p)
    return str(p)


def test_load_roi_products_returns_raw_class_map(tmp_path):
    src = _cm()
    mp = _write_mask(tmp_path, src)
    w, b, lb, cm, fo = camera_utils.load_roi_products(
        mp, (6, 6), (6, 6), 0, LUT, "fail_open", want_class_map=True)
    assert fo is False and lb is None and cm is not None
    assert cm.dtype == torch.uint8
    assert torch.equal(cm.cpu(), src)                 # raw, undilated ids
    assert w is not None and b is not None


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


@pytest.mark.parametrize("cw,sr,expected", [
    ("", "", False),                    # both off -> byte-identical old path
    (None, None, False),
    ("2:2.0", "", True),                # densify weighting alone (pre-existing)
    ("", "2:0.01", True),               # NEW: scale reg alone must request the map
    (None, "2:0.01", True),
    ("2:2.0", "2:0.01", True),
])
def test_loadcam_want_class_map_condition(cw, sr, expected):
    expr = _want_class_map_expr()
    assert "roi_scale_reg" in expr and "roi_densify_class_weights" in expr
    args = SimpleNamespace(roi_densify_class_weights=cw, roi_scale_reg=sr)
    assert eval(expr, {"args": args}) is expected


def test_loadcam_want_class_map_condition_defaults_off():
    # Missing attributes entirely (a namespace built before either flag existed).
    assert eval(_want_class_map_expr(), {"args": SimpleNamespace()}) is False


# ---------------------------------------------------------------------------
# Off-path: nothing runs, nothing parses
# ---------------------------------------------------------------------------

def test_loss_site_guard_is_inert_when_scale_reg_is_off(monkeypatch):
    # Replica of train.py's guard. Both sentinels None (flag off, or on but before
    # the first attribution refresh) -> loss untouched, no attribution/render work.
    def boom(*_a, **_k):
        raise AssertionError("attribution ran on the off-path")

    monkeypatch.setattr(fast_utils, "render_fastgs", boom)
    monkeypatch.setattr(fast_utils, "attribute_gaussians_by_class", boom)

    for scale_reg_groups, scale_reg_attr in [(None, None),
                                             ([(0.01, [2])], None),
                                             (None, torch.zeros(3, dtype=torch.int8))]:
        loss = torch.tensor(0.125)
        before = float(loss)
        if scale_reg_groups is not None and scale_reg_attr is not None:
            loss = loss + 999.0                       # unreachable in these cases
        assert float(loss) == before


def test_train_py_parses_spec_only_when_nonempty():
    # The empty default must never reach parse_scale_reg_spec (which raises on "").
    with open(os.path.join(_REPO_ROOT, "train.py"), "r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    guard_idx = next(i for i, ln in enumerate(lines) if ln.strip() == "if scale_reg_spec:")
    guard_indent = len(lines[guard_idx]) - len(lines[guard_idx].lstrip())
    call_idx = next(i for i, ln in enumerate(lines)
                    if "parse_scale_reg_spec(scale_reg_spec)" in ln)
    call_indent = len(lines[call_idx]) - len(lines[call_idx].lstrip())
    assert call_idx > guard_idx
    assert call_indent > guard_indent                  # inside the "if spec:" block
    # and the spec itself defaults to the empty string
    assert re.search(r'scale_reg_spec = str\(getattr\(dataset, "roi_scale_reg", ""\) or ""\)',
                     "\n".join(lines))
