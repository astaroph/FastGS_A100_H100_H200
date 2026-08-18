import torch
from PIL import ImageFilter
from gaussian_renderer import render_fastgs
from .loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
import torchvision.transforms as transforms
import random


def sampling_cameras(my_viewpoint_stack, requested=10):
    """Randomly sample up to `requested` cameras from the viewpoint stack.

    Clamped to the stack size so smoke tests / sparse scenes with fewer than
    `requested` views do not crash. Default preserves the original behavior
    (10 cameras, same RNG consumption) on normal scenes.
    """
    num_cams = min(max(int(requested), 1), len(my_viewpoint_stack))
    camlist = []
    for _ in range(num_cams):
        loc = random.randint(0, len(my_viewpoint_stack) - 1)
        camlist.append(my_viewpoint_stack.pop(loc))
    return camlist


def get_raw_l1_map(reconstructed_image, original_image):
    """Per-pixel channel-mean L1 map, detached. Shape (H, W)."""
    return torch.mean(torch.abs(reconstructed_image - original_image), 0).detach()


def normalize_map_full(l1_loss_map):
    """Min-max normalize a per-pixel map over the FULL frame (baseline behavior)."""
    denom = torch.max(l1_loss_map) - torch.min(l1_loss_map)
    if torch.abs(denom) < 1e-12:
        return torch.zeros_like(l1_loss_map)
    return (l1_loss_map - torch.min(l1_loss_map)) / denom


def normalize_map_roi(l1_loss_map, roi_mask):
    """Min-max normalize a per-pixel map using statistics from ROI pixels only.

    Full-frame normalization lets a deliberately under-supervised background's
    residual dominate the frame max, compressing normalized ROI errors toward 0
    and silently starving ROI densification; normalizing within the ROI keeps
    loss_thresh's meaning stable. Values outside the ROI are normalized with the
    same (ROI-derived) statistics — callers AND the result with the ROI anyway.
    """
    vals = l1_loss_map[roi_mask]
    if vals.numel() == 0:
        return torch.zeros_like(l1_loss_map)
    mn = torch.min(vals)
    denom = torch.max(vals) - mn
    if torch.abs(denom) < 1e-12:
        return torch.zeros_like(l1_loss_map)
    return (l1_loss_map - mn) / denom


def get_loss(reconstructed_image, original_image):
    return normalize_map_full(get_raw_l1_map(reconstructed_image, original_image))


def compute_photometric_loss(viewpoint_cam, image):
    gt_image = viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))
    return loss


def normalize(config_value, value_tensor):
    multiplier = config_value
    value_tensor[value_tensor.isnan()] = 0
    valid_indices = (value_tensor > 0)
    valid_value = value_tensor[valid_indices].to(torch.float32)
    ret_value = torch.zeros_like(value_tensor, dtype=torch.float32)
    if valid_value.numel() > 0:
        ret_value[valid_indices] = multiplier * (valid_value / torch.median(valid_value))
    return ret_value


def _stencil_counts(cam, gaussians, pipe, bg, args, stencil):
    """One binary get_flag render: per-gaussian int32 counts of stencil pixels
    the gaussian actually blends into (occlusion-aware). The CUDA kernel counts
    metric_map[pix] == 1 EXACTLY, so the stencil must be binary (bool/0-1)."""
    return render_fastgs(
        cam, gaussians, pipe, bg, args.mult,
        get_flag=True, metric_map=stencil.int(),
    )["accum_metric_counts"]


def attribute_gaussians_by_class(camlist, gaussians, pipe, bg, args, groups):
    """Per-gaussian lambda-group attribution + observed-direction resultant.

    For each sampled camera that carries a class map: one full-frame binary
    render per lambda group (union stencil of the group's class ids) plus one
    remainder render (every other pixel -- required so a silhouette-edge
    gaussian overhanging a regularized class is not mis-attributed to it).
    Counts are occlusion-aware via the get_flag path.

    attr[i] = index into ``groups`` when that group's total count strictly
    exceeds every other column (remainder included); -1 otherwise (tie,
    remainder win, or never rendered by the sampled cameras -- all
    conservative: no regularization).

    r_bar[i] = count-weighted mean unit direction gaussian -> camera over ALL
    columns' counts, deliberately NOT renormalized: its norm rho <= 1 encodes
    sweep diversity, so well-swept gaussians self-attenuate the scale-reg
    penalty (train.py multiplies the hinge threshold by |a1 . r_bar|). Rows
    with attr == -1 carry no meaning in r_bar.

    Fail-open cameras (roi_class_map is None) are skipped; train.py's coverage
    check bounds how many of those can exist.

    Returns (attr int8 (N,), r_bar float32 (N, 3)) on the model's device.
    """
    device = gaussians.get_xyz.device
    n = gaussians.get_xyz.shape[0]
    if len(groups) > 126:
        # attr is int8 with -1 as the unattributed sentinel; a wider group list
        # would wrap silently. Unreachable with real specs (<= 256 class ids).
        raise ValueError("attribute_gaussians_by_class supports at most 126 groups, "
                         "got {}".format(len(groups)))
    n_cols = len(groups) + 1
    totals = torch.zeros((n, n_cols), dtype=torch.float32, device=device)
    dir_acc = torch.zeros((n, 3), dtype=torch.float32, device=device)
    for cam in camlist:
        class_map = getattr(cam, "roi_class_map", None)
        if class_map is None:
            continue
        cm = class_map.to(device)
        remaining = torch.ones_like(cm, dtype=torch.bool)
        cam_counts = torch.zeros(n, dtype=torch.float32, device=device)
        for gi, (_lam, ids) in enumerate(groups):
            gmask = torch.zeros_like(remaining)
            for cid in ids:
                gmask |= (cm == int(cid))
            remaining &= ~gmask
            if not bool(gmask.any()):
                continue
            counts = _stencil_counts(cam, gaussians, pipe, bg, args, gmask).float()
            totals[:, gi] += counts
            cam_counts += counts
        if bool(remaining.any()):
            counts = _stencil_counts(cam, gaussians, pipe, bg, args, remaining).float()
            totals[:, -1] += counts
            cam_counts += counts
        cam_center = cam.camera_center.to(device).float().reshape(1, 3)
        dirs = cam_center - gaussians.get_xyz.detach()
        dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)
        dir_acc += cam_counts[:, None] * dirs
    wsum = totals.sum(dim=1)
    r_bar = dir_acc / wsum.clamp_min(1.0)[:, None]
    best_vals, best_idx = totals.max(dim=1)
    ties = (totals == best_vals[:, None]).sum(dim=1) > 1
    attr = best_idx.to(torch.int8)
    attr[(best_idx == (n_cols - 1)) | ties | (best_vals <= 0)] = -1
    return attr, r_bar


def _class_weighted_counts(cam, gaussians, pipe, bg, args, dmap, groups):
    """Per-gaussian densify counts with per-class weights (float result).

    Splits the flagged map into one BINARY metric map per distinct weight (ids
    sharing a weight are grouped by parse_densify_class_weights) and renders each
    through the existing get_flag path -- the CUDA kernel counts
    metric_map[pix] == 1 EXACTLY, so scaled map values would be silently dropped;
    weighting therefore happens on the counts, never inside the map. Flagged
    pixels whose class is in no group (including dilation-halo pixels whose raw
    class is background) keep weight 1.0 via the remainder render. Requires
    cam.roi_class_map (train.py validates coverage at startup).
    """
    class_map = getattr(cam, "roi_class_map", None)
    if class_map is None:
        raise RuntimeError(
            "[ROI-DENSIFY-CW] camera {!r} has no roi_class_map; startup validation "
            "should have caught this".format(getattr(cam, "image_name", "?")))
    cm = class_map.to(dmap.device)
    remaining = dmap > 0
    weighted = None
    for weight, ids in groups:
        gmask = torch.zeros_like(remaining)
        for cid in ids:
            gmask |= (cm == int(cid))
        sel = remaining & gmask
        remaining = remaining & ~gmask
        if weight == 0.0:
            # Pixels are consumed (excluded from the weight-1.0 remainder) but a
            # render whose counts get multiplied by zero is pure waste.
            continue
        if not bool(sel.any()):
            continue
        counts = _stencil_counts(cam, gaussians, pipe, bg, args, sel)
        contrib = counts.float() * float(weight)
        weighted = contrib if weighted is None else weighted + contrib
    if bool(remaining.any()):
        counts = _stencil_counts(cam, gaussians, pipe, bg, args, remaining)
        weighted = counts.float() if weighted is None else weighted + counts.float()
    if weighted is None:
        weighted = torch.zeros(
            gaussians.get_xyz.shape[0], dtype=torch.float32, device=dmap.device)
    return weighted


def compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, args, DENSIFY=False, roi_cfg=None):
    """
    Compute multi-view consistency scores for Gaussians to guide densification/pruning.
    Hardened against empty models and zero-variance score tensors.

    ROI extension (roi_cfg=None reproduces baseline behavior exactly, including a
    single flagged render per camera). roi_cfg is a namespace with:
      densify_active (bool)  - gate densification counts to each camera's ROI
      densify_mode (str)     - "intersect" | "blend"
      bg_scale (float)       - blend mode: fractional influence of out-of-ROI
                               high-error pixels (missed-foreground recovery)
      track_roi_touch (bool) - also count, per gaussian, how many ROI pixels it
                               touches (object-only background pruning evidence)

    The pruning score is ALWAYS computed from the unmasked full-frame high-error
    map — ROI must never rewrite pruning behavior (a shared map would make
    background permanently prune-immune and turn the late prune into an
    ROI-targeted prune). Fail-open cameras (roi_bin is None) contribute a ZERO
    densification map — they train unmasked but must not re-open background
    densification — and are skipped for roi_touch (they can neither confirm nor
    deny background status).

    Returns (importance_score, pruning_score, roi_touch); roi_touch is None
    unless tracked and at least one sampled camera had a usable ROI. When
    roi_cfg is set, roi_cfg.last_flagged_px_mean is updated with the mean
    flagged densify-map pixel count per sampled view (densification-starvation
    diagnostic).
    """
    n = gaussians.get_xyz.shape[0]
    device = gaussians.get_xyz.device if n > 0 else torch.device("cuda")

    if n == 0 or len(camlist) == 0:
        empty = torch.zeros((0,), dtype=torch.float32, device=device)
        return (empty if DENSIFY else None), empty, None

    roi_active = roi_cfg is not None and getattr(roi_cfg, "densify_active", False)
    track_touch = roi_cfg is not None and getattr(roi_cfg, "track_roi_touch", False)
    cw_groups = getattr(roi_cfg, "class_weight_groups", None) if roi_cfg is not None else None

    densify_counts = None
    blend_outside_counts = None
    full_metric_score = None
    roi_touch = None
    flagged_px_total = 0.0
    flagged_px_views = 0

    for view in range(len(camlist)):
        my_viewpoint_cam = camlist[view]

        render_image = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, args.mult)["render"]
        photometric_loss = compute_photometric_loss(my_viewpoint_cam, render_image)

        gt_image = my_viewpoint_cam.original_image.cuda()
        raw_map = get_raw_l1_map(render_image, gt_image)
        prune_map = (normalize_map_full(raw_map) > args.loss_thresh).int()

        # Full-frame counts: byte-identical to baseline; always feed the pruning
        # score, and also feed densification when ROI is inactive.
        render_pkg = render_fastgs(
            my_viewpoint_cam,
            gaussians,
            pipe,
            bg,
            args.mult,
            get_flag=True,
            metric_map=prune_map,
        )
        prune_counts = render_pkg["accum_metric_counts"]

        if full_metric_score is None:
            full_metric_score = photometric_loss * prune_counts.clone()
        else:
            full_metric_score += photometric_loss * prune_counts

        if DENSIFY:
            roi_bin = getattr(my_viewpoint_cam, "roi_bin", None)
            if not roi_active:
                d_counts = prune_counts
            elif roi_bin is None:
                # fail-open view: zero densification contribution. Float when class
                # weighting is on so accumulation dtype stays consistent with the
                # weighted counts from masked views.
                if cw_groups:
                    d_counts = torch.zeros_like(prune_counts, dtype=torch.float32)
                else:
                    d_counts = torch.zeros_like(prune_counts)
                if roi_cfg.densify_mode == "blend":
                    o_counts = torch.zeros_like(prune_counts)
            else:
                roi_mask = roi_bin.to(raw_map.device) > 0
                dmap = ((normalize_map_roi(raw_map, roi_mask) > args.loss_thresh) & roi_mask).int()
                flagged_px_total += float(dmap.sum().item())
                flagged_px_views += 1
                if cw_groups:
                    d_counts = _class_weighted_counts(
                        my_viewpoint_cam, gaussians, pipe, bg, args, dmap, cw_groups)
                else:
                    d_counts = render_fastgs(
                        my_viewpoint_cam, gaussians, pipe, bg, args.mult,
                        get_flag=True, metric_map=dmap,
                    )["accum_metric_counts"]
                if roi_cfg.densify_mode == "blend":
                    # Deliberate deviation from the plan's count-subtraction formula:
                    # dmap is ROI-locally normalized while prune_map is full-frame
                    # normalized, so (fullframe - roi) subtraction is ill-defined
                    # (dmap is not a subset of prune_map). The outside channel is
                    # rendered directly instead; dmap and outside_map are disjoint
                    # by construction (dmap subset of roi, outside_map subset of ~roi).
                    outside_map = (prune_map.bool() & ~roi_mask).int()
                    o_counts = render_fastgs(
                        my_viewpoint_cam, gaussians, pipe, bg, args.mult,
                        get_flag=True, metric_map=outside_map,
                    )["accum_metric_counts"]

            if densify_counts is None:
                densify_counts = d_counts.clone()
            else:
                densify_counts += d_counts
            if roi_active and roi_cfg.densify_mode == "blend":
                if blend_outside_counts is None:
                    blend_outside_counts = o_counts.clone()
                else:
                    blend_outside_counts += o_counts

        if track_touch:
            roi_bin = getattr(my_viewpoint_cam, "roi_bin", None)
            if roi_bin is not None:
                touch_map = (roi_bin.to(device) > 0).int()
                t_counts = render_fastgs(
                    my_viewpoint_cam, gaussians, pipe, bg, args.mult,
                    get_flag=True, metric_map=touch_map,
                )["accum_metric_counts"]
                if roi_touch is None:
                    roi_touch = t_counts.clone()
                else:
                    roi_touch += t_counts

    if roi_cfg is not None:
        roi_cfg.last_flagged_px_mean = (
            flagged_px_total / flagged_px_views if flagged_px_views > 0 else None
        )

    if full_metric_score is None or full_metric_score.numel() == 0:
        empty = torch.zeros((0,), dtype=torch.float32, device=device)
        return (empty if DENSIFY else None), empty, roi_touch

    score_min = torch.min(full_metric_score)
    score_max = torch.max(full_metric_score)
    denom = score_max - score_min

    if torch.abs(denom) < 1e-12:
        pruning_score = torch.zeros_like(full_metric_score, dtype=torch.float32)
    else:
        pruning_score = (full_metric_score - score_min) / denom

    if DENSIFY:
        if densify_counts is None or densify_counts.numel() == 0:
            importance_score = torch.zeros((0,), dtype=torch.float32, device=device)
        elif roi_active and roi_cfg.densify_mode == "blend":
            outside = blend_outside_counts if blend_outside_counts is not None else torch.zeros_like(densify_counts)
            importance_score = (
                densify_counts.float() + float(roi_cfg.bg_scale) * outside.float()
            ) / float(len(camlist))
        elif roi_active and cw_groups:
            # Class-weighted counts are float; floor division would zero the
            # fractional weights, so use true division (blend-branch precedent).
            importance_score = densify_counts.float() / float(len(camlist))
        else:
            importance_score = torch.div(densify_counts, len(camlist), rounding_mode="floor")
    else:
        importance_score = None

    return importance_score, pruning_score, roi_touch
