#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import numpy as np
from PIL import Image
from scene.cameras import Camera
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False

# ROI (FastGS + LightSeg): per-process LUT cache, keyed by the raw --roi_class_weights spec
# string, so repeated Scene loads in one process (e.g. train.py then render.py in the same
# interpreter) don't re-parse it. See
# docs/FASTGS_ROI_lightseg_implementation_plan_2026-08-06.md §4.3 item 2.
_ROI_LUT_CACHE = {}


def _get_roi_lut(class_weights_spec):
    lut = _ROI_LUT_CACHE.get(class_weights_spec)
    if lut is None:
        import utils.roi_utils as roi_utils  # deferred: only required when ROI is actually used
        lut = roi_utils.parse_class_weights(class_weights_spec)
        _ROI_LUT_CACHE[class_weights_spec] = lut
    return lut


def load_roi_products(mask_path, expected_size, resolution, dilate_px, lut, missing_policy):
    """Load a per-view class-ID mask PNG and turn it into ROI training products.

    :param mask_path: path to the class-ID PNG written by the mask exporter.
    :param expected_size: (w, h) the mask is expected to match — the source image's PIL
        ``.size`` (i.e. ``cam_info.image.size``), checked before any resizing.
    :param resolution: (w, h) to NEAREST-resize the class map to — the same resolution the
        RGB image was loaded at.
    :param dilate_px: dilation radius in *source-image* pixels; scaled to ``resolution``.
    :param lut: 256-entry class-weight LUT (``roi_utils.parse_class_weights`` output).
    :param missing_policy: ``"fail_open"`` or ``"fail_loud"``.
    :return: ``(weight_map, roi_bin, failopen)`` — ``weight_map`` is a (1,H,W) fp16 tensor or
        None, ``roi_bin`` is a (H,W) uint8 tensor or None, ``failopen`` is bool. On fail-open,
        both tensors are None.
    """
    import utils.roi_utils as roi_utils  # deferred: only required when ROI is actually used

    def _fail(reason):
        msg = "[ROI] fail-open: {} (mask_path={})".format(reason, mask_path)
        if missing_policy == "fail_loud":
            raise RuntimeError("[ROI] fail-loud: {} (mask_path={})".format(reason, mask_path))
        print(msg)
        return None, None, True

    if not mask_path or not os.path.isfile(mask_path):
        return _fail("mask file missing or not a file")

    try:
        pil_mask = Image.open(mask_path)
        pil_mask.load()
    except Exception as e:
        return _fail("could not open mask ({})".format(e))

    # Only mode "L" (the exporter's format) and "P" (palette: np.array yields the raw
    # indices, which ARE the class ids) carry class IDs faithfully. convert("L") on
    # any other mode would map colors to LUMINANCE values (class 2 -> ~76 etc.), which
    # the LUT defaults to weight 1.0 — silent corruption instead of a policy failure.
    if pil_mask.mode not in ("L", "P"):
        return _fail("unsupported mask mode {!r} (expected 'L' class-ID PNG)".format(pil_mask.mode))

    if tuple(pil_mask.size) != tuple(expected_size):
        return _fail("mask size {} != expected image size {}".format(pil_mask.size, tuple(expected_size)))

    # NEVER PILtoTorch here — its resize has no resample arg (defaults to bicubic) and would
    # invent fractional class IDs. NEAREST is the only resample that preserves the class-ID
    # value set exactly.
    pil_mask = pil_mask.resize(resolution, Image.NEAREST)
    class_map = torch.from_numpy(np.array(pil_mask, dtype=np.uint8))

    eff_d = 0
    if dilate_px > 0:
        eff_d = int(round(dilate_px * (resolution[0] / expected_size[0])))
        eff_d = max(1, eff_d)

    weight_map, roi_bin = roi_utils.build_roi_tensors(class_map, lut, eff_d)
    return weight_map, roi_bin, False


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    # ROI (FastGS + LightSeg): only touched when this CameraInfo actually carries a mask path
    # (i.e. Scene was built with roi_for_training=True and --use_roi_masks). Empty mask_path
    # (the default) leaves roi_kwargs empty so the Camera(...) call below is byte-identical to
    # the pre-ROI code path.
    roi_kwargs = {}
    if cam_info.mask_path:
        lut = _get_roi_lut(args.roi_class_weights)
        weight_map, roi_bin, failopen = load_roi_products(
            cam_info.mask_path, cam_info.image.size, resolution, args.roi_dilate_px, lut, args.roi_missing)
        roi_kwargs = dict(roi_weight=weight_map, roi_bin=roi_bin, roi_failopen=failopen,
                           mask_relpath=cam_info.mask_relpath)

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  **roi_kwargs)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
