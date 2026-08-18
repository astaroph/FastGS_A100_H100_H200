"""Pure, unit-testable helpers for FastGS ROI masks (LightSeg-derived class maps).

No CUDA-extension imports at module level: `masked_ssim` lazily imports `fused_ssim`
so this module always imports cleanly even on machines without the compiled
extension. Stick to torch 1.12-compatible APIs only (F.max_pool2d, tensor indexing,
torch.where, .half()/.float(), clamp_min) -- both the A100 (torch 1.12) and
H100/H200 (torch 2.4) cluster envs must run this file unmodified.
"""

import math

import torch
import torch.nn.functional as F


def parse_class_weights(spec):
    """Parse a comma-separated "id:weight" spec into a 256-entry float32 CPU LUT.

    spec: e.g. "0:0.15,1:1.0,2:1.0,3:1.0,4:1.0" -- comma-separated "id:weight" pairs.
    Returns: (256,) float32 CPU tensor; lut[class_id] -> weight. Class ids not listed
    default to 1.0 (fail-safe: unknown classes get full supervision, never silent 0).

    Raises ValueError (with a message naming the offending pair) on: empty spec / no
    valid entries, a pair that isn't exactly "id:weight", a non-integer id, an id
    outside [0, 255], a duplicate id, or a weight that isn't finite or outside
    [0.0, 1.0].
    """
    if spec is None or not spec.strip():
        raise ValueError("roi class-weight spec is empty: {!r}".format(spec))

    lut = torch.ones(256, dtype=torch.float32)
    seen_ids = set()
    n_valid = 0

    for raw_pair in spec.split(","):
        pair = raw_pair.strip()
        if not pair:
            continue

        parts = pair.split(":")
        if len(parts) != 2:
            raise ValueError(
                "malformed class-weight pair {!r} in spec {!r} (expected 'id:weight')".format(pair, spec)
            )
        id_str, weight_str = parts[0].strip(), parts[1].strip()

        try:
            class_id = int(id_str)
        except ValueError:
            raise ValueError("class id {!r} in pair {!r} is not an integer".format(id_str, pair))

        try:
            weight = float(weight_str)
        except ValueError:
            raise ValueError("weight {!r} in pair {!r} is not a float".format(weight_str, pair))

        if not (0 <= class_id <= 255):
            raise ValueError("class id {} in pair {!r} is out of range [0, 255]".format(class_id, pair))
        if class_id in seen_ids:
            raise ValueError("duplicate class id {} in spec {!r}".format(class_id, spec))
        if not math.isfinite(weight):
            raise ValueError("weight {} for class id {} is not finite".format(weight, class_id))
        if not (0.0 <= weight <= 1.0):
            raise ValueError("weight {} for class id {} is out of range [0.0, 1.0]".format(weight, class_id))

        lut[class_id] = weight
        seen_ids.add(class_id)
        n_valid += 1

    if n_valid == 0:
        raise ValueError("no valid class-weight entries parsed from spec {!r}".format(spec))

    return lut


def parse_scale_reg_spec(spec):
    """Parse a comma-separated "id:lambda" spec into scale-reg lambda groups.

    spec: e.g. "2:0.01,0:0.01" -- class ids as in the mask PNGs; lambda is the
    per-class loss weight of the ray-modulated anisotropy hinge (train.py).
    Unlike parse_densify_class_weights, lambda 1.0 is meaningful and KEPT;
    entries with lambda exactly 0.0 are dropped (an explicitly disabled class).

    Returns: list of (lam, [class_ids]) groups -- ids sharing a lambda are
    grouped so attribution renders one union stencil per distinct lambda, not
    one per class. Group order is deterministic (first appearance of each
    lambda).

    Raises ValueError on: empty spec / no valid entries, malformed pairs,
    non-integer or out-of-range [0,255] ids, duplicate ids, non-finite or
    negative lambdas, or a spec whose every entry is 0.0 (a fully-disabled
    spec is a config mistake, not a feature).
    """
    if spec is None or not str(spec).strip():
        raise ValueError("scale-reg spec is empty: {!r}".format(spec))

    seen_ids = set()
    groups = []          # list of (lam, [ids]) in first-appearance order
    by_lam = {}
    n_valid = 0
    for raw_pair in str(spec).split(","):
        pair = raw_pair.strip()
        if not pair:
            continue
        parts = pair.split(":")
        if len(parts) != 2:
            raise ValueError(
                "malformed scale-reg pair {!r} in spec {!r} (expected 'id:lambda')".format(pair, spec))
        try:
            class_id = int(parts[0].strip())
        except ValueError:
            raise ValueError("class id {!r} in pair {!r} is not an integer".format(parts[0], pair))
        try:
            lam = float(parts[1].strip())
        except ValueError:
            raise ValueError("lambda {!r} in pair {!r} is not a float".format(parts[1], pair))
        if not (0 <= class_id <= 255):
            raise ValueError("class id {} in pair {!r} is out of range [0, 255]".format(class_id, pair))
        if class_id in seen_ids:
            raise ValueError("duplicate class id {} in spec {!r}".format(class_id, spec))
        if not math.isfinite(lam) or lam < 0.0:
            raise ValueError("lambda {} for class id {} must be finite and >= 0".format(lam, class_id))
        seen_ids.add(class_id)
        n_valid += 1
        if lam == 0.0:
            continue
        if lam in by_lam:
            by_lam[lam].append(class_id)
        else:
            by_lam[lam] = [class_id]
            groups.append((lam, by_lam[lam]))
    if n_valid == 0:
        raise ValueError("scale-reg spec {!r} has no valid entries".format(spec))
    if not groups:
        raise ValueError(
            "scale-reg spec {!r} is fully disabled (every lambda is 0.0); omit the "
            "flag instead".format(spec))
    if len(groups) > 126:
        # attribution stores the group index in int8 with -1 as the sentinel;
        # reject at parse time so misconfiguration fails before Scene load.
        raise ValueError(
            "scale-reg spec has {} distinct lambdas; at most 126 are supported".format(len(groups)))
    return [(lam, ids) for lam, ids in groups]


def parse_densify_class_weights(spec):
    """Parse a comma-separated "id:weight" spec into densify weight groups.

    spec: e.g. "2:2.0,3:2.0,1:0.5" -- class ids as in the mask PNGs. Unlike
    parse_class_weights, weights may exceed 1.0 (they scale densification
    importance counts, not loss weights); ids not listed implicitly weight 1.0.

    Returns: list of (weight, [class_ids]) groups -- ids sharing a weight are
    grouped so the densify pass renders one metric map per distinct weight, not
    per class. Group order is deterministic (first appearance of each weight).
    Entries with weight exactly 1.0 are dropped (identity; the unlisted-classes
    remainder already renders at weight 1.0).

    Raises ValueError on: empty spec / no valid entries, malformed pairs,
    non-integer or out-of-range [0,255] ids, duplicate ids, non-finite or
    negative weights, or a spec whose every entry is 1.0 (a fully-identity spec
    is a config mistake, not a feature).
    """
    if spec is None or not str(spec).strip():
        raise ValueError("densify class-weight spec is empty: {!r}".format(spec))

    seen_ids = set()
    groups = []          # list of [weight, [ids]] in first-appearance order
    by_weight = {}
    n_valid = 0
    for raw_pair in str(spec).split(","):
        pair = raw_pair.strip()
        if not pair:
            continue
        parts = pair.split(":")
        if len(parts) != 2:
            raise ValueError(
                "malformed densify class-weight pair {!r} in spec {!r} (expected 'id:weight')".format(pair, spec))
        try:
            class_id = int(parts[0].strip())
        except ValueError:
            raise ValueError("class id {!r} in pair {!r} is not an integer".format(parts[0], pair))
        try:
            weight = float(parts[1].strip())
        except ValueError:
            raise ValueError("weight {!r} in pair {!r} is not a float".format(parts[1], pair))
        if not (0 <= class_id <= 255):
            raise ValueError("class id {} in pair {!r} is out of range [0, 255]".format(class_id, pair))
        if class_id in seen_ids:
            raise ValueError("duplicate class id {} in spec {!r}".format(class_id, spec))
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("weight {} for class id {} must be finite and >= 0".format(weight, class_id))
        seen_ids.add(class_id)
        n_valid += 1
        if weight == 1.0:
            continue  # identity entry: same as leaving the class unlisted
        if weight not in by_weight:
            by_weight[weight] = []
            groups.append([weight, by_weight[weight]])
        by_weight[weight].append(class_id)

    if n_valid == 0:
        raise ValueError("no valid entries parsed from densify class-weight spec {!r}".format(spec))
    if not groups:
        raise ValueError(
            "densify class-weight spec {!r} is entirely weight-1.0 entries (an identity "
            "spec); leave the flag unset instead".format(spec))
    return [(w, list(ids)) for w, ids in groups]


def build_roi_tensors(class_map, lut, dilate_px, label_scale=1.0, label_class_id=-1,
                      return_label_bin=False):
    """Build the per-pixel weight map and binary ROI stencil from a class-id map.

    class_map: (H,W) uint8 tensor of class ids. lut: 256-entry float32 tensor from
    parse_class_weights (any device). dilate_px: int >= 0, Chebyshev dilation radius
    in pixels, shared by both outputs.

    Optional label extensions (both default-off; the defaults reproduce the original
    two-tuple outputs bit-for-bit):
      label_scale / label_class_id: when label_scale != 1.0 and label_class_id >= 0,
        the LUT weight of exactly that class is multiplied by label_scale BEFORE the
        dilation-max, so label halos inherit the scaled weight (per-view clarity
        scalar s_v, FASTGS_ROI_VIEW_WEIGHTING). The scaled value may exceed 1.0 --
        parse_class_weights' [0,1] clamp governs the base LUT only.
      return_label_bin: also return a (H,W) uint8 {0,1} stencil of the label class,
        dilated with the SAME kernel as roi_bin (late label refinement,
        FASTGS_ROI_LATE_LABEL_REFINE). Requires label_class_id >= 0.

    Returns (weight_map, roi_bin), or (weight_map, roi_bin, label_bin) when
    return_label_bin is True:
      weight_map: (1,H,W) float16. lut[class_map] grayscale-dilated (each pixel takes
        the max class weight within the Chebyshev radius, so halos inherit the
        strongest neighboring class weight), computed in float32 then cast to half.
      roi_bin: (H,W) uint8 in {0,1}. Dilated (class_map > 0) -- any foreground class --
        using the SAME kernel. Deliberately independent of the weight values: changing
        --roi_class_weights must never change what counts as ROI/background.

    All computation happens in float32 (CPU fp16 max_pool2d is unsupported on old
    torch); output device matches class_map's device.
    """
    if dilate_px < 0:
        raise ValueError("dilate_px must be >= 0, got {}".format(dilate_px))
    if return_label_bin and label_class_id < 0:
        raise ValueError("return_label_bin=True requires label_class_id >= 0")
    if label_scale != 1.0 and label_class_id < 0:
        raise ValueError("label_scale != 1.0 requires label_class_id >= 0")
    if not math.isfinite(label_scale) or label_scale < 0.0:
        raise ValueError("label_scale must be finite and >= 0, got {}".format(label_scale))

    lut = lut.to(device=class_map.device, dtype=torch.float32)
    class_ids_long = class_map.long()

    weight0 = lut[class_ids_long]                      # (H,W) float32
    fg0 = (class_map > 0).to(torch.float32)             # (H,W) float32

    label0 = None
    if label_class_id >= 0 and (label_scale != 1.0 or return_label_bin):
        label_mask = class_ids_long == int(label_class_id)
        if label_scale != 1.0:
            # Pre-dilation so the halo's max-pool inherits the scaled label weight.
            weight0 = torch.where(label_mask, weight0 * float(label_scale), weight0)
        if return_label_bin:
            label0 = label_mask.to(torch.float32)

    def _dilate(x2d):
        # Separable Chebyshev dilation: two 1-D max pools are mathematically
        # identical to one (2d+1)x(2d+1) pool for max, and ~14x faster. The full
        # 2-D pool cost ~30s/view single-threaded at 3800x2533 (2 maps/view x
        # every training view = the dominant scene-load cost on 1-CPU cluster
        # jobs); callers should also pass a CUDA class_map when available
        # (~86 ms/view including upload).
        if dilate_px == 0:
            return x2d[None]
        k = 2 * dilate_px + 1
        return F.max_pool2d(
            F.max_pool2d(x2d[None], kernel_size=(1, k), stride=1, padding=(0, dilate_px)),
            kernel_size=(k, 1), stride=1, padding=(dilate_px, 0))

    weight_f32 = _dilate(weight0)
    fg_f32 = _dilate(fg0)

    if label_class_id >= 0 and label_scale != 1.0:
        # Post-dilation exactness restore. The shared max-dilation lets a
        # HIGHER-weighted neighbor within dilate_px overwrite a DOWN-scaled
        # (label_scale < 1, e.g. renormalized unclear views) label pixel's own
        # weight — total erosion for labels narrower than 2*dilate_px. Halo
        # pixels keep max semantics (an up-scaled label still bleeds outward);
        # the label's OWN pixels always end at exactly lut[label] * label_scale
        # (a no-op numerically when label_scale >= 1, where the max already
        # picked the scaled value).
        label_val = float(lut[int(label_class_id)].item()) * float(label_scale)
        weight_f32 = torch.where(
            label_mask[None], torch.full_like(weight_f32, label_val), weight_f32)

    weight_map = weight_f32.half()                       # (1,H,W) float16
    roi_bin = (fg_f32[0] > 0).to(torch.uint8)             # (H,W) uint8 in {0,1}

    if return_label_bin:
        label_bin = (_dilate(label0)[0] > 0).to(torch.uint8)  # (H,W) uint8 in {0,1}
        return weight_map, roi_bin, label_bin
    return weight_map, roi_bin


def _resolve_denom(weight, h, w, norm):
    """Shared denominator rule for masked_l1 / masked_ssim."""
    if norm == "roi":
        return 3.0 * weight.sum().clamp_min(1e-8)
    if norm == "global":
        return 3.0 * h * w
    raise ValueError("unknown norm {!r}: expected 'roi' or 'global'".format(norm))


def masked_l1(image, gt, weight, norm):
    """Weighted mean absolute error.

    image, gt: (3,H,W) float32. weight: (1,H,W) float32. norm: "roi" normalizes by
    3*sum(weight) (weighted-pixel count); "global" normalizes by 3*H*W (full-frame
    pixel count, same denominator regardless of weight).

    With weight all-ones and norm="roi" this equals torch.abs(image - gt).mean()
    exactly (fp32) -- both denominator variants reduce to the plain-mean case.
    Raises ValueError on an unrecognized norm.
    """
    h, w = image.shape[-2], image.shape[-1]
    denom = _resolve_denom(weight, h, w, norm)
    return (torch.abs(image - gt) * weight).sum() / denom


def masked_ssim(image, gt, weight, norm):
    """Weighted mean of the per-pixel fused-SSIM map.

    image, gt: (1,3,H,W) float32. weight: (1,H,W) float32, unsqueezed internally to
    (1,1,H,W) so it broadcasts across the channel dim. norm: same denominator rule as
    masked_l1 ("roi" -> 3*sum(weight), "global" -> 3*H*W).

    Lazily imports fused_ssim.FusedSSIMMap (the compiled CUDA extension is not
    available on every dev machine) so importing this module never requires it;
    raises RuntimeError naming fused_ssim if the import fails at call time.

    With weight all-ones and norm="roi" this equals fused_ssim(image, gt) exactly.
    """
    try:
        from fused_ssim import FusedSSIMMap
    except ImportError as exc:
        raise RuntimeError(
            "masked_ssim requires the 'fused_ssim' package (compiled CUDA extension); "
            "import failed: {}".format(exc)
        )

    h, w = image.shape[-2], image.shape[-1]
    denom = _resolve_denom(weight, h, w, norm)

    ssim_map = FusedSSIMMap.apply(0.01 ** 2, 0.03 ** 2, image, gt, "same", True)  # (1,3,H,W)
    w_bcast = weight.unsqueeze(0)  # (1,1,H,W)

    return (ssim_map * w_bcast).sum() / denom
