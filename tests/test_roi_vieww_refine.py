"""Tests for the two default-off label-attention arms (2026-08-17):

  - per-view clarity scalar (FASTGS_ROI_VIEW_WEIGHTING): build_roi_tensors label_scale
    pre-dilation + camera_utils roi_view_weights.json lookup (fail-loud contract);
  - late label refinement (FASTGS_ROI_LATE_LABEL_REFINE): label_bin stencil plumbing.

The load-bearing assertions are the byte-identity ones: with the new arguments left at
their defaults, every output must be bit-identical to the pre-change behavior.
"""
import json

import numpy as np
import pytest
import torch
from PIL import Image

from utils import camera_utils, roi_utils


LUT = roi_utils.parse_class_weights("0:0.0,1:0.3,2:1.0,3:1.0,4:1.0")


def _class_map():
    # 12x12: background frame, mount block left, label patch center-right, specimen row.
    m = torch.zeros((12, 12), dtype=torch.uint8)
    m[2:10, 1:4] = 1    # mount
    m[4:7, 6:9] = 2     # label
    m[9, 5:11] = 3      # specimen
    return m


# ---------------------------------------------------------------------------
# build_roi_tensors: defaults are byte-identical
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dilate", [0, 2])
def test_defaults_bit_identical(dilate):
    m = _class_map()
    w_ref, b_ref = roi_utils.build_roi_tensors(m, LUT, dilate)
    w_new, b_new = roi_utils.build_roi_tensors(
        m, LUT, dilate, label_scale=1.0, label_class_id=-1, return_label_bin=False)
    assert torch.equal(w_ref, w_new)
    assert torch.equal(b_ref, b_new)
    # scale exactly 1.0 with a valid class id is also an identity
    w_id, b_id = roi_utils.build_roi_tensors(m, LUT, dilate, label_scale=1.0, label_class_id=2)
    assert torch.equal(w_ref, w_id)
    assert torch.equal(b_ref, b_id)


# ---------------------------------------------------------------------------
# build_roi_tensors: label_scale semantics
# ---------------------------------------------------------------------------

def test_label_scale_scales_only_label_class_no_dilation():
    m = _class_map()
    w, b = roi_utils.build_roi_tensors(m, LUT, 0, label_scale=1.7, label_class_id=2)
    w = w[0].float()
    assert torch.allclose(w[m == 2], torch.tensor(1.7), atol=1e-3)
    assert torch.allclose(w[m == 1], torch.tensor(0.3), atol=1e-3)
    assert torch.allclose(w[m == 3], torch.tensor(1.0), atol=1e-3)
    assert torch.allclose(w[m == 0], torch.tensor(0.0), atol=1e-3)
    # roi_bin must be untouched by the scale (weight-independent by contract)
    _, b_ref = roi_utils.build_roi_tensors(m, LUT, 0)
    assert torch.equal(b, b_ref)


def test_label_scale_applies_before_dilation_max():
    # A pixel adjacent to the label patch must inherit the SCALED label weight
    # through the max-dilation (halo semantics).
    m = _class_map()
    w, _ = roi_utils.build_roi_tensors(m, LUT, 1, label_scale=2.0, label_class_id=2)
    w = w[0].float()
    # (4,5) is background immediately left of the label patch at (4,6)
    assert m[4, 5].item() == 0
    assert w[4, 5].item() == pytest.approx(2.0, abs=1e-2)


def test_label_scale_may_exceed_one():
    m = _class_map()
    w, _ = roi_utils.build_roi_tensors(m, LUT, 0, label_scale=2.0, label_class_id=2)
    assert float(w.float().max().item()) == pytest.approx(2.0, abs=1e-2)


def test_label_downweight_survives_dilation_next_to_higher_neighbor():
    # Adversarial-review finding: the shared max-dilation used to ERASE a
    # down-scaled (s < 1, renormalized unclear view) label weight wherever a
    # higher-weighted class sits within dilate_px â€” 100% erosion for labels
    # narrower than 2*dilate_px. The post-dilation exactness restore must keep
    # the label's OWN pixels at exactly lut[label] * s.
    m = torch.zeros((12, 12), dtype=torch.uint8)
    m[4:7, 3:6] = 2   # small label (3 px wide, well under 2*dilate_px)
    m[4:7, 6:9] = 3   # specimen (weight 1.0) directly adjacent
    w, _ = roi_utils.build_roi_tensors(m, LUT, 2, label_scale=0.5, label_class_id=2)
    w = w[0].float()
    assert torch.allclose(w[m == 2], torch.tensor(0.5), atol=1e-3), \
        "down-scaled label pixels were rescued by the dilation max"
    # the specimen keeps its own weight; the down-scale never bleeds outward
    assert torch.allclose(w[m == 3], torch.tensor(1.0), atol=1e-3)
    # and the up-scale direction still bleeds outward through the halo
    w_up, _ = roi_utils.build_roi_tensors(m, LUT, 2, label_scale=2.0, label_class_id=2)
    assert torch.allclose(w_up[0].float()[m == 2], torch.tensor(2.0), atol=1e-2)


@pytest.mark.parametrize("bad", [float("nan"), -0.5, float("inf")])
def test_label_scale_invalid_raises(bad):
    with pytest.raises(ValueError):
        roi_utils.build_roi_tensors(_class_map(), LUT, 0, label_scale=bad, label_class_id=2)


def test_label_args_coherence_raises():
    with pytest.raises(ValueError):
        roi_utils.build_roi_tensors(_class_map(), LUT, 0, label_scale=1.5)  # no class id
    with pytest.raises(ValueError):
        roi_utils.build_roi_tensors(_class_map(), LUT, 0, return_label_bin=True)  # no class id


# ---------------------------------------------------------------------------
# build_roi_tensors: label_bin stencil
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dilate", [0, 2])
def test_label_bin_matches_dilated_label_mask(dilate):
    m = _class_map()
    w3, b3, lb = roi_utils.build_roi_tensors(
        m, LUT, dilate, label_class_id=2, return_label_bin=True)
    # the 2-tuple outputs are unchanged by asking for the stencil
    w2, b2 = roi_utils.build_roi_tensors(m, LUT, dilate)
    assert torch.equal(w3, w2)
    assert torch.equal(b3, b2)
    assert lb.dtype == torch.uint8
    if dilate == 0:
        assert torch.equal(lb.bool(), m == 2)
    else:
        # every label pixel is covered, and the stencil never exceeds the
        # Chebyshev-dilated bounding region of the label patch
        assert bool(lb.bool()[m == 2].all())
        ys, xs = torch.nonzero(m == 2, as_tuple=True)
        y0, y1 = ys.min().item() - dilate, ys.max().item() + dilate
        x0, x1 = xs.min().item() - dilate, xs.max().item() + dilate
        outside = torch.ones_like(lb, dtype=torch.bool)
        outside[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1] = False
        assert not bool(lb.bool()[outside].any())


# ---------------------------------------------------------------------------
# camera_utils: roi_view_weights.json lookup contract
# ---------------------------------------------------------------------------

def _write_vieww(tmp_path, per_view, schema_version=1, label_class_id=2):
    p = tmp_path / "roi_view_weights.json"
    p.write_text(json.dumps({
        "schema_version": schema_version,
        "label_class_id": label_class_id,
        "per_view": [{"relpath": k, "s_label": v} for k, v in per_view.items()],
    }))
    return str(p)


def test_lookup_view_scale_happy_path(tmp_path):
    jp = _write_vieww(tmp_path, {"cam3_x/img_001.JPG": 1.8, "cam1_x/img_002.JPG": 1.0})
    s, cid = camera_utils._lookup_view_scale(jp, "cam3_x/img_001.JPG")
    assert s == pytest.approx(1.8)
    assert cid == 2
    # backslash-form relpath resolves to the same key
    s2, _ = camera_utils._lookup_view_scale(jp, "cam3_x\\img_001.JPG")
    assert s2 == pytest.approx(1.8)


def test_lookup_view_scale_missing_file_raises(tmp_path):
    with pytest.raises(RuntimeError, match="file missing"):
        camera_utils._lookup_view_scale(str(tmp_path / "nope.json"), "a/b.JPG")


def test_lookup_view_scale_unknown_schema_raises(tmp_path):
    jp = _write_vieww(tmp_path, {"a/b.JPG": 1.0}, schema_version=99)
    with pytest.raises(RuntimeError, match="schema_version"):
        camera_utils._lookup_view_scale(jp, "a/b.JPG")


def test_lookup_view_scale_missing_relpath_raises(tmp_path):
    jp = _write_vieww(tmp_path, {"a/b.JPG": 1.0})
    with pytest.raises(RuntimeError, match="not present"):
        camera_utils._lookup_view_scale(jp, "a/OTHER.JPG")


def test_lookup_view_scale_bad_value_raises(tmp_path):
    jp = _write_vieww(tmp_path, {"a/b.JPG": float("nan")})
    with pytest.raises(RuntimeError, match="bad s_label"):
        camera_utils._lookup_view_scale(jp, "a/b.JPG")


# ---------------------------------------------------------------------------
# load_roi_products: 4-tuple contract
# ---------------------------------------------------------------------------

def _write_mask(tmp_path, arr):
    p = tmp_path / "mask.png"
    Image.fromarray(arr.numpy().astype(np.uint8), mode="L").save(p)
    return str(p)


def test_load_roi_products_failopen_returns_four_nones_and_flag(tmp_path):
    w, b, lb, fo = camera_utils.load_roi_products(
        str(tmp_path / "missing.png"), (12, 12), (12, 12), 0, LUT, "fail_open")
    assert w is None and b is None and lb is None and fo is True


def test_load_roi_products_default_matches_direct_build(tmp_path):
    m = _class_map()
    mp = _write_mask(tmp_path, m)
    w, b, lb, fo = camera_utils.load_roi_products(mp, (12, 12), (12, 12), 0, LUT, "fail_open")
    assert fo is False and lb is None
    w_ref, b_ref = roi_utils.build_roi_tensors(m.to(w.device), LUT, 0)
    assert torch.equal(w.cpu(), w_ref.cpu())
    assert torch.equal(b.cpu(), b_ref.cpu())


def test_load_roi_products_scale_and_stencil(tmp_path):
    m = _class_map()
    mp = _write_mask(tmp_path, m)
    w, b, lb, fo = camera_utils.load_roi_products(
        mp, (12, 12), (12, 12), 0, LUT, "fail_open",
        label_scale=1.5, label_class_id=2, want_label_bin=True)
    assert fo is False
    assert lb is not None and torch.equal(lb.cpu().bool(), m == 2)
    assert float(w.cpu().float()[0][m == 2].max().item()) == pytest.approx(1.5, abs=1e-2)
