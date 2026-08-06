import math

import pytest
import torch

from utils import roi_utils


# ---------------------------------------------------------------------------
# parse_class_weights
# ---------------------------------------------------------------------------

def test_parse_class_weights_defaults_and_listed_values():
    lut = roi_utils.parse_class_weights("0:0.15,1:1.0,2:1.0,3:1.0,4:1.0")
    assert lut.shape == (256,)
    assert lut.dtype == torch.float32
    assert lut[0].item() == pytest.approx(0.15)
    assert lut[1].item() == pytest.approx(1.0)
    assert lut[2].item() == pytest.approx(1.0)
    assert lut[3].item() == pytest.approx(1.0)
    assert lut[4].item() == pytest.approx(1.0)
    # unlisted ids default to 1.0 (fail-safe)
    assert lut[5].item() == pytest.approx(1.0)
    assert lut[255].item() == pytest.approx(1.0)


def test_parse_class_weights_sparse_spec_defaults_unlisted_to_one():
    lut = roi_utils.parse_class_weights("2:0.4")
    assert lut[2].item() == pytest.approx(0.4)
    for cid in (0, 1, 3, 100, 255):
        assert lut[cid].item() == pytest.approx(1.0)


def test_parse_class_weights_tolerates_whitespace():
    lut = roi_utils.parse_class_weights(" 0:0.2 , 1 : 0.9 ")
    assert lut[0].item() == pytest.approx(0.2)
    assert lut[1].item() == pytest.approx(0.9)


@pytest.mark.parametrize("spec", ["", "   ", ",,,", None])
def test_parse_class_weights_empty_spec_raises(spec):
    with pytest.raises(ValueError):
        roi_utils.parse_class_weights(spec)


@pytest.mark.parametrize(
    "spec",
    [
        "1",              # no colon
        "1:2:3",          # too many colons
        "1:0.5,",         # trailing comma is fine, but keep a malformed sibling
        "abc:0.5",        # non-integer id
        "1:abc",          # non-float weight
        "1:0.5:extra",    # too many colons variant
    ],
)
def test_parse_class_weights_malformed_pair_raises(spec):
    if spec == "1:0.5,":
        # trailing comma alone is valid (empty trailing token is skipped);
        # this parametrize entry exists to document that fact, not to raise.
        lut = roi_utils.parse_class_weights(spec)
        assert lut[1].item() == pytest.approx(0.5)
        return
    with pytest.raises(ValueError):
        roi_utils.parse_class_weights(spec)


def test_parse_class_weights_id_out_of_range_raises():
    with pytest.raises(ValueError):
        roi_utils.parse_class_weights("256:0.5")
    with pytest.raises(ValueError):
        roi_utils.parse_class_weights("-1:0.5")


def test_parse_class_weights_duplicate_id_raises():
    with pytest.raises(ValueError):
        roi_utils.parse_class_weights("0:0.1,0:0.2")


@pytest.mark.parametrize("bad_weight", ["1.5", "-0.1", "nan", "inf", "-inf"])
def test_parse_class_weights_bad_weight_raises(bad_weight):
    with pytest.raises(ValueError):
        roi_utils.parse_class_weights("0:{}".format(bad_weight))


def test_parse_class_weights_boundary_weights_are_valid():
    lut = roi_utils.parse_class_weights("0:0.0,1:1.0")
    assert lut[0].item() == pytest.approx(0.0)
    assert lut[1].item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# build_roi_tensors
# ---------------------------------------------------------------------------

def test_build_roi_tensors_dilate_zero_is_identity():
    torch.manual_seed(0)
    class_map = torch.randint(0, 5, (9, 11), dtype=torch.uint8)
    lut = roi_utils.parse_class_weights("0:0.15,1:0.5,2:1.0,3:0.7,4:1.0")

    weight_map, roi_bin = roi_utils.build_roi_tensors(class_map, lut, dilate_px=0)

    assert weight_map.shape == (1, 9, 11)
    assert weight_map.dtype == torch.float16
    assert roi_bin.shape == (9, 11)
    assert roi_bin.dtype == torch.uint8
    assert set(roi_bin.unique().tolist()).issubset({0, 1})

    expected_weight = lut[class_map.long()].half()
    expected_roi_bin = (class_map > 0).to(torch.uint8)
    assert torch.equal(weight_map[0], expected_weight)
    assert torch.equal(roi_bin, expected_roi_bin)


def _make_single_point_class_map(size, center, fg_id):
    class_map = torch.zeros((size, size), dtype=torch.uint8)
    class_map[center, center] = fg_id
    return class_map


def test_build_roi_tensors_dilation_halo_is_exact_chebyshev_radius():
    size = 21
    center = 10
    dilate_px = 3
    class_map = _make_single_point_class_map(size, center, fg_id=2)
    lut = roi_utils.parse_class_weights("0:0.15,2:1.0")

    weight_map, roi_bin = roi_utils.build_roi_tensors(class_map, lut, dilate_px=dilate_px)

    # weight_map is stored as float16; allow for its ~1e-3 rounding error, not
    # exact fp32 equality.
    w = weight_map[0].float()
    for r in range(size):
        for c in range(size):
            cheby = max(abs(r - center), abs(c - center))
            if cheby <= dilate_px:
                assert w[r, c].item() == pytest.approx(1.0, abs=2e-3), (r, c, cheby)
                assert roi_bin[r, c].item() == 1, (r, c, cheby)
            else:
                assert w[r, c].item() == pytest.approx(0.15, abs=2e-3), (r, c, cheby)
                assert roi_bin[r, c].item() == 0, (r, c, cheby)


def test_build_roi_tensors_roi_bin_independent_of_weights():
    size = 21
    class_map = _make_single_point_class_map(size, 10, fg_id=1)  # e.g. "mount" class
    lut_a = roi_utils.parse_class_weights("0:0.15,1:1.0")
    lut_b = roi_utils.parse_class_weights("0:0.15,1:0.4")

    _, roi_bin_a = roi_utils.build_roi_tensors(class_map, lut_a, dilate_px=3)
    weight_map_b, roi_bin_b = roi_utils.build_roi_tensors(class_map, lut_b, dilate_px=3)

    assert torch.equal(roi_bin_a, roi_bin_b)
    # but the weight maps DO differ (sanity check the two LUTs actually disagree);
    # weight_map is float16, so allow for its rounding error.
    assert weight_map_b[0].max().item() == pytest.approx(0.4, abs=2e-3)


def test_build_roi_tensors_multiclass_grayscale_dilation_takes_max_neighbor():
    # two adjacent foreground points with different weights; the halo between them
    # should inherit the STRONGER (max) neighboring weight, not an average/nearest one.
    size = 15
    class_map = torch.zeros((size, size), dtype=torch.uint8)
    class_map[7, 4] = 1   # weight 0.4
    class_map[7, 10] = 2  # weight 1.0
    lut = roi_utils.parse_class_weights("0:0.15,1:0.4,2:1.0")

    weight_map, _ = roi_utils.build_roi_tensors(class_map, lut, dilate_px=3)
    w = weight_map[0].float()

    # pixel (7,7) is within Chebyshev distance 3 of BOTH points; must take the max (1.0)
    assert w[7, 7].item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# masked_l1
# ---------------------------------------------------------------------------

def test_masked_l1_all_ones_roi_norm_equals_plain_mean_exactly():
    torch.manual_seed(1)
    image = torch.rand(3, 8, 8, dtype=torch.float32)
    gt = torch.rand(3, 8, 8, dtype=torch.float32)
    weight = torch.ones(1, 8, 8, dtype=torch.float32)

    loss = roi_utils.masked_l1(image, gt, weight, norm="roi")
    plain = torch.abs(image - gt).mean()

    assert torch.equal(loss, plain)


def test_masked_l1_all_ones_global_norm_equals_plain_mean_exactly():
    torch.manual_seed(2)
    image = torch.rand(3, 8, 8, dtype=torch.float32)
    gt = torch.rand(3, 8, 8, dtype=torch.float32)
    weight = torch.ones(1, 8, 8, dtype=torch.float32)

    loss = roi_utils.masked_l1(image, gt, weight, norm="global")
    plain = torch.abs(image - gt).mean()

    assert torch.equal(loss, plain)


def test_masked_l1_gradient_flows_and_is_zero_where_weight_is_zero():
    torch.manual_seed(3)
    h, w = 6, 6
    image = torch.rand(3, h, w, dtype=torch.float32, requires_grad=True)
    gt = torch.rand(3, h, w, dtype=torch.float32)

    weight = torch.zeros(1, h, w, dtype=torch.float32)
    weight[:, :, : w // 2] = 1.0  # left half in-ROI, right half weight 0

    loss = roi_utils.masked_l1(image, gt, weight, norm="roi")
    loss.backward()

    assert image.grad is not None
    zero_region = image.grad[:, :, w // 2 :]
    assert torch.equal(zero_region, torch.zeros_like(zero_region))
    nonzero_region = image.grad[:, :, : w // 2]
    assert nonzero_region.abs().sum().item() > 0.0


def test_masked_l1_unknown_norm_raises():
    image = torch.rand(3, 4, 4)
    gt = torch.rand(3, 4, 4)
    weight = torch.ones(1, 4, 4)
    with pytest.raises(ValueError):
        roi_utils.masked_l1(image, gt, weight, norm="bogus")


# ---------------------------------------------------------------------------
# masked_ssim (skips cleanly if fused_ssim is not installed on this machine)
# ---------------------------------------------------------------------------

def test_masked_ssim_module_import_does_not_require_fused_ssim():
    # roi_utils itself must import fine even without fused_ssim installed --
    # already exercised implicitly by every test in this file importing roi_utils
    # at module scope, but assert it explicitly for clarity.
    assert hasattr(roi_utils, "masked_ssim")


def test_masked_ssim_missing_dependency_raises_clear_runtime_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "fused_ssim":
            raise ImportError("No module named 'fused_ssim'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    image = torch.rand(1, 3, 8, 8)
    gt = torch.rand(1, 3, 8, 8)
    weight = torch.ones(1, 8, 8)

    with pytest.raises(RuntimeError, match="fused_ssim"):
        roi_utils.masked_ssim(image, gt, weight, norm="roi")


def test_masked_ssim_all_ones_roi_norm_equals_fused_ssim():
    # fused_ssim is a CUDA-only kernel: skip without the extension AND without a GPU.
    pytest.importorskip("fused_ssim")
    if not torch.cuda.is_available():
        pytest.skip("fused_ssim requires CUDA tensors")
    from fused_ssim import fused_ssim

    torch.manual_seed(4)
    image = torch.rand(1, 3, 32, 32, dtype=torch.float32, device="cuda")
    gt = torch.rand(1, 3, 32, 32, dtype=torch.float32, device="cuda")
    weight = torch.ones(1, 32, 32, dtype=torch.float32, device="cuda")

    loss = roi_utils.masked_ssim(image, gt, weight, norm="roi")
    plain = fused_ssim(image, gt)

    assert torch.equal(loss, plain)


def test_masked_ssim_gradient_flows():
    pytest.importorskip("fused_ssim")
    if not torch.cuda.is_available():
        pytest.skip("fused_ssim requires CUDA tensors")

    torch.manual_seed(5)
    image = torch.rand(1, 3, 16, 16, dtype=torch.float32, device="cuda", requires_grad=True)
    gt = torch.rand(1, 3, 16, 16, dtype=torch.float32, device="cuda")
    weight = torch.zeros(1, 16, 16, dtype=torch.float32, device="cuda")
    weight[:, :, :8] = 1.0

    loss = roi_utils.masked_ssim(image, gt, weight, norm="roi")
    loss.backward()

    assert image.grad is not None


def test_masked_ssim_unknown_norm_raises():
    pytest.importorskip("fused_ssim")
    if not torch.cuda.is_available():
        pytest.skip("fused_ssim requires CUDA tensors")

    image = torch.rand(1, 3, 8, 8, device="cuda")
    gt = torch.rand(1, 3, 8, 8, device="cuda")
    weight = torch.ones(1, 8, 8, device="cuda")
    with pytest.raises(ValueError):
        roi_utils.masked_ssim(image, gt, weight, norm="bogus")
