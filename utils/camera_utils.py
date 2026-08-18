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

import json
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

# Per-view clarity scalars (FASTGS_ROI_VIEW_WEIGHTING): per-process cache of
# roi_view_weights.json, keyed by absolute path. The file is written by the
# pipeline's stats tool (tools/roi_view_weights.py) next to the mask dir and
# joined to cameras by mask relpath — the same relpath key roi_manifest.json
# uses, so no camera-name parsing happens on this side.
_ROI_VIEW_WEIGHTS_CACHE = {}


def _get_roi_lut(class_weights_spec):
    lut = _ROI_LUT_CACHE.get(class_weights_spec)
    if lut is None:
        import utils.roi_utils as roi_utils  # deferred: only required when ROI is actually used
        lut = roi_utils.parse_class_weights(class_weights_spec)
        _ROI_LUT_CACHE[class_weights_spec] = lut
    return lut


def _get_roi_view_weights(json_path):
    """Load and cache roi_view_weights.json.

    Returns (per_view: dict relpath->s_label, label_class_id: int). Fail-LOUD on a
    missing/unreadable/unknown-schema file: the flag being set means the stats tool
    was supposed to have run this very job — a silent identity here would be
    indistinguishable from the feature working (the §9 'never silent all-ones' rule).
    """
    entry = _ROI_VIEW_WEIGHTS_CACHE.get(json_path)
    if entry is not None:
        return entry
    if not os.path.isfile(json_path):
        raise RuntimeError(
            "[ROI-VIEWW] --roi_view_weights_json set but file missing: {}".format(json_path))
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as exc:
        raise RuntimeError(
            "[ROI-VIEWW] could not parse {}: {}".format(json_path, exc))
    if data.get("schema_version") != 1:
        raise RuntimeError(
            "[ROI-VIEWW] unknown schema_version {!r} in {} (expected 1)".format(
                data.get("schema_version"), json_path))
    label_class_id = int(data["label_class_id"])
    per_view = {}
    for pv in data["per_view"]:
        s = float(pv["s_label"])
        if not (s == s) or s < 0.0:  # NaN / negative guard without importing math
            raise RuntimeError(
                "[ROI-VIEWW] bad s_label {!r} for {!r} in {}".format(
                    pv.get("s_label"), pv.get("relpath"), json_path))
        per_view[str(pv["relpath"]).replace("\\", "/")] = s
    entry = (per_view, label_class_id)
    _ROI_VIEW_WEIGHTS_CACHE[json_path] = entry
    return entry


def _lookup_view_scale(json_path, mask_relpath):
    """s_label + label_class_id for one camera; fail-LOUD on a missing relpath key.

    A key miss means the stats tool and the scene disagree about the view list —
    weighting an unknown subset silently is exactly the mis-weighting hazard the
    plan's §9 table forbids.
    """
    per_view, label_class_id = _get_roi_view_weights(json_path)
    key = str(mask_relpath).replace("\\", "/")
    if key not in per_view:
        raise RuntimeError(
            "[ROI-VIEWW] mask relpath {!r} not present in {} ({} entries)".format(
                key, json_path, len(per_view)))
    return per_view[key], label_class_id


def load_roi_products(mask_path, expected_size, resolution, dilate_px, lut, missing_policy,
                      label_scale=1.0, label_class_id=-1, want_label_bin=False,
                      want_class_map=False):
    """Load a per-view class-ID mask PNG and turn it into ROI training products.

    :param mask_path: path to the class-ID PNG written by the mask exporter.
    :param expected_size: (w, h) the mask is expected to match — the source image's PIL
        ``.size`` (i.e. ``cam_info.image.size``), checked before any resizing.
    :param resolution: (w, h) to NEAREST-resize the class map to — the same resolution the
        RGB image was loaded at.
    :param dilate_px: dilation radius in *source-image* pixels; scaled to ``resolution``.
    :param lut: 256-entry class-weight LUT (``roi_utils.parse_class_weights`` output).
    :param missing_policy: ``"fail_open"`` or ``"fail_loud"``.
    :param label_scale / label_class_id / want_label_bin: passed through to
        ``build_roi_tensors`` (per-view clarity scalar and/or late-refinement label
        stencil); the defaults reproduce the original behavior bit-for-bit.
    :param want_class_map: also return the UNdilated resized class-ID map (uint8, raw
        class ids) — class-scoped densify weighting (FASTGS_ROI_DENSIFY_CLASS_WEIGHTS)
        partitions flagged pixels by class at densify time.
    :return: ``(weight_map, roi_bin, label_bin, class_map, failopen)`` — ``weight_map``
        is a (1,H,W) fp16 tensor or None; ``roi_bin``/``label_bin``/``class_map`` are
        (H,W) uint8 tensors or None (``label_bin`` only when ``want_label_bin``,
        ``class_map`` only when ``want_class_map``); ``failopen`` is bool. On
        fail-open, all tensors are None.
    """
    import utils.roi_utils as roi_utils  # deferred: only required when ROI is actually used

    def _fail(reason):
        msg = "[ROI] fail-open: {} (mask_path={})".format(reason, mask_path)
        if missing_policy == "fail_loud":
            raise RuntimeError("[ROI] fail-loud: {} (mask_path={})".format(reason, mask_path))
        print(msg)
        return None, None, None, None, True

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
    if torch.cuda.is_available():
        # Build the dilated products on GPU: ~86 ms/view vs seconds/view on a
        # single-CPU cluster allocation. Camera.__init__ moves the results to
        # data_device afterwards, so residency is unchanged.
        class_map = class_map.cuda()

    eff_d = 0
    if dilate_px > 0:
        eff_d = int(round(dilate_px * (resolution[0] / expected_size[0])))
        eff_d = max(1, eff_d)

    if want_label_bin:
        weight_map, roi_bin, label_bin = roi_utils.build_roi_tensors(
            class_map, lut, eff_d, label_scale=label_scale,
            label_class_id=label_class_id, return_label_bin=True)
    else:
        weight_map, roi_bin = roi_utils.build_roi_tensors(
            class_map, lut, eff_d, label_scale=label_scale, label_class_id=label_class_id)
        label_bin = None
    return weight_map, roi_bin, label_bin, (class_map if want_class_map else None), False


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
        # Per-view clarity scalar (FASTGS_ROI_VIEW_WEIGHTING): joined by mask relpath,
        # fail-loud on a missing file or key. Default "" leaves label_scale at 1.0 and
        # label_class_id at -1 so the build below is bit-identical to the flag-off path.
        label_scale = 1.0
        label_class_id = -1
        vieww_json = str(getattr(args, "roi_view_weights_json", "") or "")
        if vieww_json:
            label_scale, label_class_id = _lookup_view_scale(vieww_json, cam_info.mask_relpath)
        # Late label refinement (FASTGS_ROI_LATE_LABEL_REFINE): keep the dilated label
        # stencil on the camera so train.py can boost label pixels post-densification.
        want_label_bin = bool(getattr(args, "roi_keep_label_bin", False))
        if want_label_bin and label_class_id < 0:
            label_class_id = int(getattr(args, "roi_label_class_id", 2))
        # Class-scoped densify weighting: keep the raw class map on the camera so the
        # densify pass can partition flagged pixels by class. "" = off, zero cost.
        want_class_map = bool(str(getattr(args, "roi_densify_class_weights", "") or ""))
        weight_map, roi_bin, label_bin, class_map, failopen = load_roi_products(
            cam_info.mask_path, cam_info.image.size, resolution, args.roi_dilate_px, lut,
            args.roi_missing, label_scale=label_scale, label_class_id=label_class_id,
            want_label_bin=want_label_bin, want_class_map=want_class_map)
        roi_kwargs = dict(roi_weight=weight_map, roi_bin=roi_bin, roi_failopen=failopen,
                           mask_relpath=cam_info.mask_relpath, roi_label_bin=label_bin,
                           roi_class_map=class_map)

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
