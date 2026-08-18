"""Tests for class-scoped densify weighting (FASTGS_ROI_DENSIFY_CLASS_WEIGHTS) and
the solidity dials (2026-08-17).

_class_weighted_counts is exercised with a monkeypatched render_fastgs so the
grouping / weighting / remainder logic is verified CPU-only, without the compiled
rasterizer. The load-bearing assertions are again the byte-identity ones: defaults
must reproduce pre-change behavior exactly.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

# CPU-only collection stubs, self-contained (never rely on sibling test files having
# collected first — default alphabetical order collects THIS file first). Each stub is
# LOUD: reaching the real functionality fails the test instead of silently passing.
import sys
import types

try:
    import simple_knn  # noqa: F401
except ImportError:
    # Same stub pattern as test_roi_loading.py: required only by
    # scene.gaussian_model's distCUDA2, which these tests never exercise.
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


# ---------------------------------------------------------------------------
# parse_densify_class_weights
# ---------------------------------------------------------------------------

def test_parse_groups_by_weight_first_appearance_order():
    groups = roi_utils.parse_densify_class_weights("2:2.0,3:2.0,1:0.5")
    assert groups == [(2.0, [2, 3]), (0.5, [1])]


def test_parse_drops_identity_entries():
    groups = roi_utils.parse_densify_class_weights("2:2.0,4:1.0")
    assert groups == [(2.0, [2])]


def test_parse_weights_may_exceed_one():
    groups = roi_utils.parse_densify_class_weights("2:8.5")
    assert groups == [(8.5, [2])]


@pytest.mark.parametrize("bad", [
    "", "   ", None,                 # empty
    "2",                              # no colon
    "2:1:0",                          # extra colon
    "x:2.0",                          # non-int id
    "2:abc",                          # non-float weight
    "300:2.0",                        # id out of range
    "2:2.0,2:3.0",                    # duplicate id
    "2:-1.0",                         # negative weight
    "2:inf",                          # non-finite
    "2:1.0,3:1.0",                    # fully-identity spec
])
def test_parse_rejects(bad):
    with pytest.raises(ValueError):
        roi_utils.parse_densify_class_weights(bad)


# ---------------------------------------------------------------------------
# _class_weighted_counts (render_fastgs monkeypatched)
# ---------------------------------------------------------------------------

def _cm():
    # 6x6: rows 0-1 class 1 (mount), rows 2-3 class 2 (label), row 4 class 3
    # (specimen), row 5 class 0 (background / halo remainder inside dmap).
    m = torch.zeros((6, 6), dtype=torch.uint8)
    m[0:2, :] = 1
    m[2:4, :] = 2
    m[4, :] = 3
    return m


def _fake_render(calls):
    def fake(cam, gaussians, pipe, bg, mult, get_flag=None, metric_map=None):
        assert get_flag is True
        # .int() is int32 on every supported torch; anything else is a regression.
        assert metric_map.dtype == torch.int32
        n_px = int(metric_map.sum().item())
        calls.append(n_px)
        n = gaussians.get_xyz.shape[0]
        # every gaussian "touches" all selected pixels: counts = n_px each
        return {"accum_metric_counts": torch.full((n,), n_px, dtype=torch.int32)}
    return fake


def test_class_weighted_counts_hand_computed(monkeypatch):
    calls = []
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_render(calls))
    cm = _cm()
    dmap = torch.ones((6, 6), dtype=torch.int)      # everything flagged
    cam = SimpleNamespace(roi_class_map=cm, image_name="t")
    gaussians = SimpleNamespace(get_xyz=torch.zeros((4, 3)))
    groups = [(2.0, [2, 3]), (0.5, [1])]
    out = fast_utils._class_weighted_counts(cam, gaussians, None, None,
                                            SimpleNamespace(mult=0.5), dmap, groups)
    # group {2,3}: 18 px at w2.0 -> 36 ; group {1}: 12 px at w0.5 -> 6 ;
    # remainder (class 0): 6 px at w1.0 -> 6 ; total 48 per gaussian, float
    assert out.dtype == torch.float32
    assert torch.allclose(out, torch.full((4,), 48.0))
    assert sorted(calls) == [6, 12, 18]             # three renders, correct partitions


def test_class_weighted_counts_respects_dmap_and_skips_empty_groups(monkeypatch):
    calls = []
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_render(calls))
    cm = _cm()
    dmap = torch.zeros((6, 6), dtype=torch.int)
    dmap[2:4, 0:3] = 1                               # only 6 label px flagged
    cam = SimpleNamespace(roi_class_map=cm, image_name="t")
    gaussians = SimpleNamespace(get_xyz=torch.zeros((2, 3)))
    groups = [(3.0, [2]), (0.5, [4])]                # class 4 absent -> no render
    out = fast_utils._class_weighted_counts(cam, gaussians, None, None,
                                            SimpleNamespace(mult=0.5), dmap, groups)
    assert torch.allclose(out, torch.full((2,), 18.0))   # 6 px * 3.0
    assert calls == [6]                              # exactly one render, no remainder


def test_class_weighted_counts_empty_dmap_returns_zeros(monkeypatch):
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_render([]))
    cam = SimpleNamespace(roi_class_map=_cm(), image_name="t")
    gaussians = SimpleNamespace(get_xyz=torch.zeros((3, 3)))
    out = fast_utils._class_weighted_counts(cam, gaussians, None, None,
                                            SimpleNamespace(mult=0.5),
                                            torch.zeros((6, 6), dtype=torch.int),
                                            [(2.0, [2])])
    assert out.dtype == torch.float32 and torch.allclose(out, torch.zeros(3))


def test_class_weighted_counts_missing_class_map_raises(monkeypatch):
    monkeypatch.setattr(fast_utils, "render_fastgs", _fake_render([]))
    cam = SimpleNamespace(roi_class_map=None, image_name="t")
    gaussians = SimpleNamespace(get_xyz=torch.zeros((3, 3)))
    with pytest.raises(RuntimeError, match="roi_class_map"):
        fast_utils._class_weighted_counts(cam, gaussians, None, None,
                                          SimpleNamespace(mult=0.5),
                                          torch.ones((6, 6), dtype=torch.int),
                                          [(2.0, [2])])


# ---------------------------------------------------------------------------
# load_roi_products: class-map plumbing + 5-tuple defaults
# ---------------------------------------------------------------------------

LUT = roi_utils.parse_class_weights("0:0.0,1:0.3,2:1.0,3:1.0,4:1.0")


def _write_mask(tmp_path, arr):
    p = tmp_path / "mask.png"
    Image.fromarray(arr.numpy().astype(np.uint8), mode="L").save(p)
    return str(p)


def test_default_class_map_is_none(tmp_path):
    mp = _write_mask(tmp_path, _cm())
    w, b, lb, cm, fo = camera_utils.load_roi_products(mp, (6, 6), (6, 6), 0, LUT, "fail_open")
    assert fo is False and lb is None and cm is None
    # defaults still bit-identical to a direct build
    w_ref, b_ref = roi_utils.build_roi_tensors(_cm().to(w.device), LUT, 0)
    assert torch.equal(w.cpu(), w_ref.cpu()) and torch.equal(b.cpu(), b_ref.cpu())


def test_want_class_map_returns_raw_ids(tmp_path):
    src = _cm()
    mp = _write_mask(tmp_path, src)
    w, b, lb, cm, fo = camera_utils.load_roi_products(
        mp, (6, 6), (6, 6), 0, LUT, "fail_open", want_class_map=True)
    assert fo is False and cm is not None
    assert cm.dtype == torch.uint8
    assert torch.equal(cm.cpu(), src)               # raw, undilated ids


def test_failopen_returns_five_nones_flag(tmp_path):
    w, b, lb, cm, fo = camera_utils.load_roi_products(
        str(tmp_path / "missing.png"), (6, 6), (6, 6), 0, LUT, "fail_open")
    assert (w, b, lb, cm) == (None, None, None, None) and fo is True
