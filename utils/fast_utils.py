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


def local_plane_normals(pts, k=16, chunk=2048):
    """Per-point local PCA normal (unit, arbitrary sign) from k nearest neighbors.

    pts: (M, 3) float tensor. Returns (M, 3) float32 normals on pts.device.
    Chunked pairwise distances keep memory at chunk x M. Used to score the
    billboard statistic |a3 . n_local| for label-attributed gaussians: a healthy
    text/paper gaussian lies flat (shortest axis ~ local surface normal); a
    billboard faces its observers instead (plan section 12 / owner observation).
    """
    m = pts.shape[0]
    if m <= k:
        raise ValueError("local_plane_normals needs more than k={} points, got {}".format(k, m))
    pts = pts.float()
    normals = torch.empty((m, 3), dtype=torch.float32, device=pts.device)
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        d2 = torch.cdist(pts[s:e], pts)              # (c, M)
        idx = d2.topk(k + 1, largest=False).indices  # includes self
        nb = pts[idx]                                # (c, k+1, 3)
        nb = nb - nb.mean(dim=1, keepdim=True)
        cov = nb.transpose(1, 2) @ nb                # (c, 3, 3)
        # smallest-eigenvector = local normal; eigh is ascending
        evecs = torch.linalg.eigh(cov).eigenvectors
        normals[s:e] = evecs[:, :, 0]
    return normals


@torch.no_grad()  # read-only diagnostics: never retain graph across the render loop
def observability_telemetry(camlist, gaussians, pipe, bg, args, label_class_id,
                            min_px=16, per_class=False, num_classes=5,
                            with_billboard=True):
    """A1 observability + billboard telemetry (READ-ONLY: no loss, no state).

    Per camera with a class map: binary stencil renders give per-gaussian counts.
    In heartbeat mode (per_class=False): 2 renders (label stencil + remainder).
    In dump mode (per_class=True): one render per class id in [0, num_classes)
    (the ids partition the map, so no remainder render is needed).

    Accumulates, with BINARY per-(gaussian, camera) support weights
    w = 1[label_count >= min_px] (raw pixel counts are footprint-dependent --
    the current geometry would otherwise feed back into its own observability
    statistic; independent-evaluation corrections #3/#4):
      M      = sum_w d d^T   (direction second moment; sign-invariant)
      r_vec  = sum_w d       (first moment; |r|/w = rho, kept as a diagnostic)
      support= sum w         (number of supporting cameras)
    plus label/total pixel counts over ALL cameras for purity (and per-class
    counts + margin in dump mode).

    Derived per gaussian (axes from build_rotation at the sorted-scale order):
      c1 = a1^T M_hat a1, c2 = a2^T M_hat a2,
      h  = sqrt(max(c1 - c2, 0))   -- preferential hiddenness of the long axis
      billboard = |a3 . n_local|   -- n_local from local_plane_normals over the
                  high-purity label pool; NaN where undefined.

    Returns a dict of per-gaussian tensors (all on the model device):
      counts (N, C float; C=2 heartbeat / num_classes dump), purity, margin
      (dump mode only, else None), support, rho, c1, c2, h, billboard,
      pool_mask (bool: purity >= 0.5 & support >= 1). NOTE pool_mask does NOT
      imply a finite billboard: pools <= K_LOCAL and rows strided out by
      POOL_CAP keep NaN — consumers must filter torch.isfinite(billboard).
      Sentinels: h/rho/c1/c2 are 0.0 where support == 0 and purity/margin are
      0.0 where never rendered — npz consumers must join on support/counts,
      not on values alone.
    """
    from utils.general_utils import build_rotation  # deferred: avoids import cycles

    device = gaussians.get_xyz.device
    n = gaussians.get_xyz.shape[0]
    n_cols = num_classes if per_class else 2
    lab_col = int(label_class_id) if per_class else 0
    counts = torch.zeros((n, n_cols), dtype=torch.float32, device=device)
    Msum = torch.zeros((n, 3, 3), dtype=torch.float32, device=device)
    rvec = torch.zeros((n, 3), dtype=torch.float32, device=device)
    support = torch.zeros(n, dtype=torch.float32, device=device)
    for cam in camlist:
        class_map = getattr(cam, "roi_class_map", None)
        if class_map is None:
            continue
        cm = class_map.to(device)
        cam_lab = None
        if per_class:
            for cid in range(num_classes):
                stencil = cm == cid
                if not bool(stencil.any()):
                    continue
                cc = _stencil_counts(cam, gaussians, pipe, bg, args, stencil).float()
                counts[:, cid] += cc
                if cid == lab_col:
                    cam_lab = cc
        else:
            stencil = cm == int(label_class_id)
            if bool(stencil.any()):
                cam_lab = _stencil_counts(cam, gaussians, pipe, bg, args, stencil).float()
                counts[:, 0] += cam_lab
            rem = ~stencil
            if bool(rem.any()):
                counts[:, 1] += _stencil_counts(cam, gaussians, pipe, bg, args, rem).float()
        if cam_lab is None:
            continue
        w = (cam_lab >= float(min_px)).float()
        if not bool(w.any()):
            continue
        cam_center = cam.camera_center.to(device).float().reshape(1, 3)
        d = cam_center - gaussians.get_xyz.detach()
        d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)
        Msum += w[:, None, None] * (d[:, :, None] * d[:, None, :])
        rvec += w[:, None] * d
        support += w
    tot = counts.sum(dim=1)
    purity = counts[:, lab_col] / tot.clamp_min(1.0)
    margin = None
    if per_class:
        others = counts.clone()
        others[:, lab_col] = -1.0
        margin = (counts[:, lab_col] - others.max(dim=1).values) / tot.clamp_min(1.0)
    ws = support.clamp_min(1.0)
    Mhat = Msum / ws[:, None, None]
    rho = (rvec / ws[:, None]).norm(dim=1)
    with torch.no_grad():
        order = torch.argsort(gaussians._scaling.detach(), dim=1, descending=True)
        R = build_rotation(gaussians._rotation.detach())
        ar = torch.arange(n, device=device)
        a1 = R[ar, :, order[:, 0]]
        a2 = R[ar, :, order[:, 1]]
        a3 = R[ar, :, order[:, 2]]
    c1 = torch.einsum("ni,nij,nj->n", a1, Mhat, a1)
    c2 = torch.einsum("ni,nij,nj->n", a2, Mhat, a2)
    h = torch.sqrt(torch.clamp(c1 - c2, min=0.0))
    billboard = torch.full((n,), float("nan"), dtype=torch.float32, device=device)
    pool = (purity >= 0.5) & (support >= 1.0)
    K_LOCAL = 16       # kNN size for local normals (local_plane_normals needs > K points)
    POOL_CAP = 60000   # cdist is O(chunk x pool): cap the kNN cloud so a bg-heavy
    #                    pool cannot balloon memory (review finding); deterministic
    #                    stride subsample, remaining pool rows keep NaN billboard.
    # with_billboard=False skips the kNN pass entirely (billboard stays all-NaN):
    # the scale-reg-v2 refresh only needs it when the plate term is armed.
    pool_idx = torch.nonzero(pool, as_tuple=True)[0]
    if with_billboard and pool_idx.numel() > K_LOCAL:
        if pool_idx.numel() > POOL_CAP:
            step = (pool_idx.numel() + POOL_CAP - 1) // POOL_CAP
            pool_idx = pool_idx[::step]
        nrm = local_plane_normals(gaussians.get_xyz.detach()[pool_idx], k=K_LOCAL)
        billboard[pool_idx] = (a3[pool_idx] * nrm).sum(dim=1).abs()
    return dict(counts=counts, purity=purity, margin=margin, support=support,
                rho=rho, c1=c1, c2=c2, h=h, billboard=billboard, pool_mask=pool)


def scale_reg_v2_penalty(scaling, h, elig1, elig2, log_r0, log_rp0, eps=1e-3,
                         with_stats=False):
    """r4 two-term scale penalty (scale-reg plan §11.5 hinge + §16/§17 plate term).

    scaling: (N, 3) log-scales WITH grad (gaussians._scaling). h / elig1 / elig2
    are detached refresh caches derived from observability_telemetry: h is the
    preferential hiddenness of the long axis; elig1 = pool & support gate;
    elig2 is additionally gated to billboard & high-h (all-False when the plate
    term is off).

    term1 (one-sided, §11 correction #5): relu(ls1 - sg(ls2) - (ln r0 - ln(h+eps))).
    Gradient reaches ls1 ONLY — the long axis shrinks toward the allowance; ls2
    is a detached target so the reg can never GROW the mid axis (the r2 flaw).
    Self-gating: h -> 0 makes the allowance huge, so weakly-hidden gaussians are
    untouched without any extra threshold.

    term2 (§16 razor plates): relu(sg(ls2) - ln rp0 - ls3). Gradient reaches ls3
    ONLY — the degenerate thin axis grows toward (mid axis / rp0); ls2 detached
    for the same one-sidedness reason.

    Means are over ELIGIBLE members (engaged or not), matching the r2 loss-site
    convention. Returns dict(term1, term2, act1, act2, n1, n2): terms are scalar
    tensors (a fresh graph-free 0.0 when the eligible set is empty); act*/n*
    (active fractions / eligible counts) are filled only under with_stats=True —
    they force GPU syncs, so the per-iteration loss site leaves them off.
    """
    device = scaling.device
    zero = torch.zeros((), dtype=scaling.dtype, device=device)
    out = dict(term1=zero, term2=zero, act1=0.0, act2=0.0, n1=0, n2=0)
    rows = elig1 | elig2
    if not bool(rows.any()):
        return out
    idx = torch.nonzero(rows, as_tuple=True)[0]
    sl = torch.sort(scaling[idx], dim=1, descending=True).values
    ls1, ls2, ls3 = sl[:, 0], sl[:, 1], sl[:, 2]
    ls2_sg = ls2.detach()
    e1 = elig1[idx]
    if bool(e1.any()):
        allow = log_r0 - torch.log(h[idx].detach() + eps)
        p1 = torch.relu(ls1 - ls2_sg - allow)[e1]
        out["term1"] = p1.mean()
        if with_stats:
            out["act1"] = float((p1 > 0).float().mean().item())
            out["n1"] = int(e1.sum().item())
    e2 = elig2[idx]
    if bool(e2.any()):
        p2 = torch.relu(ls2_sg - log_rp0 - ls3)[e2]
        out["term2"] = p2.mean()
        if with_stats:
            out["act2"] = float((p2 > 0).float().mean().item())
            out["n2"] = int(e2.sum().item())
    return out


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
