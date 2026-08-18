"""Unit tests for T3 loader plumbing (FastGS ROI masks).

Covers utils/camera_utils.py::load_roi_products / loadCam, scene/cameras.py::Camera's new
ROI kwargs, and scene/dataset_readers.py::CameraInfo's new defaulted fields. The
load_roi_products tests are CPU-only; the Camera/loadCam tests require a CUDA-capable
torch build (Camera.__init__ unconditionally .cuda()s its view/projection matrices) and
are skipped otherwise. See
docs/FASTGS_ROI_lightseg_implementation_plan_2026-08-06.md §4.1/§4.3 (mcz-3dgs-label-pipeline
repo) for the design this implements.
"""

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

# Camera.__init__ calls .cuda() on world_view_transform/projection_matrix regardless of
# data_device — the Camera/loadCam tests below cannot run on a CPU-only torch build.
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Camera.__init__ requires a CUDA-capable torch"
)

try:
    import simple_knn  # noqa: F401
except ImportError:
    # This dev box does not have the simple_knn CUDA extension built (see
    # submodules/simple-knn/ -- it needs a full MSVC+CUDA toolchain to compile on Windows).
    # It is required only by scene.gaussian_model's distCUDA2, which none of these ROI-loading
    # unit tests exercise; stub it so `scene.cameras` / `scene.dataset_readers` import cleanly
    # here. Cluster envs (§11) have the real extension built and use it unmodified.
    _simple_knn_stub = types.ModuleType("simple_knn")
    _simple_knn_c_stub = types.ModuleType("simple_knn._C")

    def _distCUDA2_stub(*_args, **_kwargs):
        raise RuntimeError("simple_knn._C.distCUDA2 stub called -- not built on this dev box")

    _simple_knn_c_stub.distCUDA2 = _distCUDA2_stub
    _simple_knn_stub._C = _simple_knn_c_stub
    sys.modules["simple_knn"] = _simple_knn_stub
    sys.modules["simple_knn._C"] = _simple_knn_c_stub

from scene.cameras import Camera
from scene.dataset_readers import CameraInfo
from utils import roi_utils
from utils.camera_utils import load_roi_products, loadCam

DEFAULT_WEIGHTS_SPEC = "0:0.15,1:1.0,2:1.0,3:1.0,4:1.0"


# ---------------------------------------------------------------------------
# load_roi_products: NEAREST resize preserves the class-ID value set
# ---------------------------------------------------------------------------

def test_nearest_resize_preserves_class_id_value_set(tmp_path):
    # 64x48 class map with 5 vertical bands, one per class id 0..4.
    w, h = 64, 48
    arr = np.zeros((h, w), dtype=np.uint8)
    band_edges = [0, 13, 26, 39, 52, w]
    for class_id, (lo, hi) in enumerate(zip(band_edges[:-1], band_edges[1:])):
        arr[:, lo:hi] = class_id
    mask_path = tmp_path / "bands.png"
    Image.fromarray(arr, mode="L").save(mask_path)

    # Distinct weight per class id so the recovered weight values reveal exactly which
    # class ids survived the resize -- if a bicubic/bilinear resize were used instead of
    # NEAREST, blended edge pixels would produce weight values outside this set.
    lut = roi_utils.parse_class_weights("0:0.0,1:0.25,2:0.5,3:0.75,4:1.0")
    resolution = (w // 2, h // 2)

    weight_map, roi_bin, _label_bin, _class_map, failopen = load_roi_products(
        str(mask_path), (w, h), resolution, 0, lut, "fail_open")

    assert not failopen
    assert weight_map.shape == (1, h // 2, w // 2)
    assert weight_map.dtype == torch.float16
    assert roi_bin.shape == (h // 2, w // 2)
    assert roi_bin.dtype == torch.uint8

    # load_roi_products builds on GPU when available (perf), so compare on CPU.
    allowed = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float16)
    for v in torch.unique(weight_map).cpu():
        assert torch.any(torch.isclose(v, allowed, atol=1e-3)), (
            "unexpected weight value {} -- NEAREST resize may have invented a new class id"
            .format(v.item())
        )


# ---------------------------------------------------------------------------
# load_roi_products: missing file / size mismatch -> fail_open vs fail_loud
# ---------------------------------------------------------------------------

def test_missing_file_fail_open_returns_none_none_true(tmp_path):
    lut = roi_utils.parse_class_weights(DEFAULT_WEIGHTS_SPEC)
    missing_path = str(tmp_path / "does_not_exist.png")

    weight_map, roi_bin, _label_bin, _class_map, failopen = load_roi_products(
        missing_path, (8, 6), (8, 6), 0, lut, "fail_open")

    assert weight_map is None
    assert roi_bin is None
    assert failopen is True


def test_missing_file_fail_loud_raises():
    lut = roi_utils.parse_class_weights(DEFAULT_WEIGHTS_SPEC)
    missing_path = "C:/definitely/not/a/real/path/mask.png"

    with pytest.raises(RuntimeError):
        load_roi_products(missing_path, (8, 6), (8, 6), 0, lut, "fail_loud")


def test_size_mismatch_fail_open_returns_none_none_true(tmp_path):
    arr = np.zeros((6, 8), dtype=np.uint8)
    mask_path = tmp_path / "mask.png"
    Image.fromarray(arr, mode="L").save(mask_path)
    lut = roi_utils.parse_class_weights(DEFAULT_WEIGHTS_SPEC)

    # mask is actually 8x6; claim the source image was 10x10 -> mismatch.
    weight_map, roi_bin, _label_bin, _class_map, failopen = load_roi_products(
        str(mask_path), (10, 10), (10, 10), 0, lut, "fail_open")

    assert weight_map is None
    assert roi_bin is None
    assert failopen is True


def test_size_mismatch_fail_loud_raises(tmp_path):
    arr = np.zeros((6, 8), dtype=np.uint8)
    mask_path = tmp_path / "mask.png"
    Image.fromarray(arr, mode="L").save(mask_path)
    lut = roi_utils.parse_class_weights(DEFAULT_WEIGHTS_SPEC)

    with pytest.raises(RuntimeError):
        load_roi_products(str(mask_path), (10, 10), (10, 10), 0, lut, "fail_loud")


# ---------------------------------------------------------------------------
# load_roi_products: dilate_px is scaled by the load-time resize factor
# ---------------------------------------------------------------------------

def test_dilate_px_scaled_to_resolution_gives_expected_halo_radius(tmp_path):
    # 80x80 source, downsampled 2x to 40x40 -> dilate_px=12 should become eff_d=6.
    expected_size = (80, 80)
    resolution = (40, 40)

    arr = np.zeros((80, 80), dtype=np.uint8)
    # Empirically verified (PIL NEAREST, 80->40) to survive downsampling as exactly one
    # destination pixel, landing at (19, 19) -- comfortably inside the frame for +/-7 probes.
    arr[39, 39] = 1
    mask_path = tmp_path / "single_fg.png"
    Image.fromarray(arr, mode="L").save(mask_path)

    lut = roi_utils.parse_class_weights("0:0.0,1:1.0")

    # dilate_px=0 baseline: locate the surviving foreground pixel at resolution scale.
    _, roi_bin_0, _lb0, _cm0, failopen0 = load_roi_products(
        str(mask_path), expected_size, resolution, 0, lut, "fail_open")
    assert not failopen0
    ys, xs = torch.nonzero(roi_bin_0, as_tuple=True)
    assert len(ys) == 1, "expected exactly one surviving foreground pixel, got {}".format(len(ys))
    r0, c0 = int(ys[0]), int(xs[0])
    assert (r0, c0) == (19, 19)
    # Margin check so the radius-7 probes below stay in bounds.
    assert 7 <= r0 <= resolution[1] - 8
    assert 7 <= c0 <= resolution[0] - 8

    # dilate_px=12 at half resolution -> eff_d = round(12 * 40/80) = 6.
    weight_map, roi_bin_12, _lb12, _cm12, failopen12 = load_roi_products(
        str(mask_path), expected_size, resolution, 12, lut, "fail_open")
    assert not failopen12
    assert weight_map.shape == (1, resolution[1], resolution[0])
    assert roi_bin_12.dtype == torch.uint8

    # Inside the radius-6 halo (Chebyshev distance <= 6 from (r0, c0)).
    assert roi_bin_12[r0, c0].item() == 1
    assert roi_bin_12[r0 + 6, c0].item() == 1
    assert roi_bin_12[r0, c0 + 6].item() == 1
    assert roi_bin_12[r0 - 6, c0].item() == 1
    assert roi_bin_12[r0, c0 - 6].item() == 1
    # Outside the halo (Chebyshev distance 7) -- proves eff_d is 6, not 7 or unscaled 12.
    assert roi_bin_12[r0 + 7, c0].item() == 0
    assert roi_bin_12[r0, c0 + 7].item() == 0
    assert roi_bin_12[r0 - 7, c0].item() == 0
    assert roi_bin_12[r0, c0 - 7].item() == 0


# ---------------------------------------------------------------------------
# Camera: new ROI kwargs stored (or default to None/False/"") correctly
# ---------------------------------------------------------------------------

@requires_cuda
def test_camera_ctor_stores_roi_products_on_cpu():
    H, W = 4, 6
    image = torch.rand(3, H, W)
    roi_weight = torch.rand(1, H, W).half()
    roi_bin = torch.randint(0, 2, (H, W), dtype=torch.uint8)

    cam = Camera(colmap_id=0, R=np.eye(3), T=np.zeros(3), FoVx=0.8, FoVy=0.6,
                 image=image, gt_alpha_mask=None, image_name="cam1_frame_0001", uid=0,
                 data_device="cpu",
                 roi_weight=roi_weight, roi_bin=roi_bin, roi_failopen=False,
                 mask_relpath="cam1/frame_0001.jpg")

    assert cam.roi_weight is not None
    assert cam.roi_weight.shape == (1, H, W)
    assert cam.roi_weight.dtype == torch.float16
    assert cam.roi_weight.device.type == "cpu"

    assert cam.roi_bin is not None
    assert cam.roi_bin.shape == (H, W)
    assert cam.roi_bin.dtype == torch.uint8
    assert cam.roi_bin.device.type == "cpu"

    assert cam.mask_relpath == "cam1/frame_0001.jpg"
    assert cam.roi_failopen is False


@requires_cuda
def test_camera_ctor_without_roi_kwargs_behaves_as_before():
    H, W = 4, 6
    image = torch.rand(3, H, W)

    cam = Camera(colmap_id=0, R=np.eye(3), T=np.zeros(3), FoVx=0.8, FoVy=0.6,
                 image=image, gt_alpha_mask=None, image_name="cam1_frame_0001", uid=0,
                 data_device="cpu")

    assert cam.roi_weight is None
    assert cam.roi_bin is None
    assert cam.roi_failopen is False
    assert cam.mask_relpath == ""


# ---------------------------------------------------------------------------
# CameraInfo: original positional construction still works (new fields defaulted)
# ---------------------------------------------------------------------------

def test_camera_info_positional_construction_fills_roi_defaults():
    ci = CameraInfo(0, np.eye(3), np.zeros(3), 0.6, 0.8, None,
                     "path/to/img.jpg", "img", 100, 80)

    assert ci.uid == 0
    assert ci.width == 100
    assert ci.height == 80
    assert ci.mask_path == ""
    assert ci.mask_relpath == ""


# ---------------------------------------------------------------------------
# loadCam: end-to-end wiring (touch points 2+3), still unit-level (no Scene)
# ---------------------------------------------------------------------------

def _fake_args(**overrides):
    base = dict(resolution=1, data_device="cpu",
                roi_class_weights=DEFAULT_WEIGHTS_SPEC, roi_dilate_px=12,
                roi_missing="fail_open")
    base.update(overrides)
    return SimpleNamespace(**base)


@requires_cuda
def test_loadcam_with_empty_mask_path_is_byte_identical_to_before():
    img = Image.new("RGB", (8, 6), color=(10, 20, 30))
    cam_info = CameraInfo(uid=0, R=np.eye(3), T=np.zeros(3), FovY=0.6, FovX=0.8,
                           image=img, image_path="unused.jpg", image_name="cam1_frame",
                           width=8, height=6)  # mask_path/mask_relpath default to ""

    cam = loadCam(_fake_args(), 0, cam_info, 1.0)

    assert cam.roi_weight is None
    assert cam.roi_bin is None
    assert cam.roi_failopen is False
    assert cam.mask_relpath == ""


@requires_cuda
def test_loadcam_with_mask_path_wires_roi_products_through(tmp_path):
    img = Image.new("RGB", (8, 6), color=(10, 20, 30))
    arr = np.zeros((6, 8), dtype=np.uint8)
    arr[:, 4:] = 1  # right half foreground
    mask_path = tmp_path / "mask.png"
    Image.fromarray(arr, mode="L").save(mask_path)

    cam_info = CameraInfo(uid=0, R=np.eye(3), T=np.zeros(3), FovY=0.6, FovX=0.8,
                           image=img, image_path="unused.jpg", image_name="cam1_frame",
                           width=8, height=6,
                           mask_path=str(mask_path), mask_relpath="cam1/frame.jpg")

    cam = loadCam(_fake_args(roi_dilate_px=0), 0, cam_info, 1.0)

    assert cam.roi_failopen is False
    assert cam.roi_weight is not None
    assert cam.roi_weight.shape == (1, 6, 8)
    assert cam.roi_bin is not None
    assert cam.roi_bin.shape == (6, 8)
    assert cam.mask_relpath == "cam1/frame.jpg"


@requires_cuda
def test_loadcam_missing_mask_fails_open_without_raising(tmp_path):
    img = Image.new("RGB", (8, 6), color=(10, 20, 30))
    missing_mask_path = str(tmp_path / "does_not_exist.png")

    cam_info = CameraInfo(uid=0, R=np.eye(3), T=np.zeros(3), FovY=0.6, FovX=0.8,
                           image=img, image_path="unused.jpg", image_name="cam1_frame",
                           width=8, height=6,
                           mask_path=missing_mask_path, mask_relpath="cam1/frame.jpg")

    cam = loadCam(_fake_args(roi_missing="fail_open"), 0, cam_info, 1.0)

    assert cam.roi_failopen is True
    assert cam.roi_weight is None
    assert cam.roi_bin is None
