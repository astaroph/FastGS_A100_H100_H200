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

import torch
import numpy as np
import math
import os, random, time, json
from random import randint
from types import SimpleNamespace
from lpipsPyTorch import lpips
from utils.loss_utils import l1_loss
from utils.roi_utils import (parse_class_weights, parse_densify_class_weights,
                             parse_scale_reg_spec, masked_l1, masked_ssim)
from fused_ssim import fused_ssim as fast_ssim
from gaussian_renderer import render_fastgs, network_gui_ws
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, build_rotation
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from utils.fast_utils import (compute_gaussian_score_fastgs, sampling_cameras,
                              attribute_gaussians_by_class, observability_telemetry,
                              scale_reg_v2_penalty)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, websockets, checkpoint_dir=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    roi_enabled = bool(getattr(dataset, "use_roi_masks", False))

    # Preemption checkpoint saves (requeue-resume plan 2026-08-28): validate
    # BEFORE any GPU-heavy work (same discipline as the ROI config checks below)
    # — an unusable configuration must fail at startup, not at the first
    # checkpoint iteration mid-training. Inert when no checkpoint iteration can
    # fire (the default is now an empty list).
    if checkpoint_iterations and any(0 < ci <= opt.iterations for ci in checkpoint_iterations):
        if not (hasattr(gaussians, "capture") and hasattr(gaussians, "restore")):
            raise RuntimeError(
                "--checkpoint_iterations requested but this GaussianModel fork lacks "
                "capture()/restore(); checkpoints could never be written or resumed.")
        # restore() unconditionally unpacks the 14-tuple that capture() only
        # produces for optimizer_type 'default' (incl. shoptimizer state); a
        # sparse_adam checkpoint would be un-restorable, so refuse to write one.
        if str(getattr(opt, "optimizer_type", "default")) != "default":
            raise RuntimeError(
                "--checkpoint_iterations requires optimizer_type 'default': restore() "
                "unpacks the 14-element capture tuple (with shoptimizer state) and "
                "would fail on the 13-element non-default capture.")

    # --- ROI config validation (BEFORE Scene: loadCam consumes roi_missing during
    # scene construction, and a typo'd enum must fail before any GPU-heavy work) --
    roi_densify_start = 0
    roi_prune_bg = False
    roi_failopen_count = 0
    roi_mean_fg_frac = None
    # Late label refinement (FASTGS_ROI_LATE_LABEL_REFINE) — resolved here, engaged in
    # the loss block only (densify/prune maps untouched by design; the densification
    # window is over before it starts).
    roi_refine = bool(getattr(opt, "roi_late_refine", False))
    roi_refine_start = 0
    roi_refine_mult = 1.0
    roi_refine_ramp = 0
    roi_vieww_json = str(getattr(dataset, "roi_view_weights_json", "") or "")
    if roi_vieww_json and not roi_enabled:
        raise RuntimeError(
            "[ROI-VIEWW] --roi_view_weights_json requires --use_roi_masks (there is no "
            "weight map to scale without ROI masks).")
    if roi_refine and not roi_enabled:
        raise RuntimeError(
            "[ROI-REFINE] --roi_late_refine requires --use_roi_masks (there is no "
            "weight map to boost without ROI masks).")
    if roi_refine and not bool(getattr(dataset, "roi_keep_label_bin", False)):
        raise RuntimeError(
            "[ROI-REFINE] --roi_late_refine requires --roi_keep_label_bin so the "
            "per-camera label stencils exist; refusing to run a silently-inert arm.")
    if roi_refine:
        roi_refine_mult = float(getattr(opt, "roi_refine_label_mult", 2.0))
        if not math.isfinite(roi_refine_mult) or roi_refine_mult < 1.0:
            raise RuntimeError(
                "[ROI-REFINE] --roi_refine_label_mult must be finite and >= 1.0, got {}".format(
                    roi_refine_mult))
        if roi_refine_mult == 1.0:
            print("[ROI-REFINE] WARNING: --roi_refine_label_mult 1.0 makes the refinement "
                  "arm a mathematical identity; it will run but change nothing.")
        start = int(getattr(opt, "roi_refine_start_iter", -1))
        roi_refine_start = start if start >= 0 else int(opt.densify_until_iter)
        roi_refine_ramp = max(0, int(getattr(opt, "roi_refine_ramp_iters", 300)))
        if roi_refine_start < int(opt.densify_until_iter):
            print("[ROI-REFINE] clamping roi_refine_start_iter {} -> {} (refinement must "
                  "not overlap the densification window; its gradients feed the densify "
                  "grad gate)".format(roi_refine_start, int(opt.densify_until_iter)))
            roi_refine_start = int(opt.densify_until_iter)
        if roi_refine_start >= int(opt.iterations):
            raise RuntimeError(
                "[ROI-REFINE] roi_refine_start_iter {} >= --iterations {}: the refinement "
                "arm could never engage this run; refusing to run a silently-inert "
                "arm.".format(roi_refine_start, int(opt.iterations)))
    # --- Solidity dials (always-on args with defaults equal to the historical
    # literals, so default runs are byte-identical; validated unconditionally) ---
    densify_min_opacity = float(getattr(opt, "densify_min_opacity", 0.005))
    final_prune_min_opacity = float(getattr(opt, "final_prune_min_opacity", 0.05))
    final_prune_score_thresh = float(getattr(opt, "final_prune_score_thresh", 0.95))
    densify_metric_gate = float(getattr(opt, "densify_metric_gate", 5.0))
    if not (0.0 < densify_min_opacity < 1.0):
        raise RuntimeError("--densify_min_opacity must be in (0,1), got {}".format(densify_min_opacity))
    # v10 collapse guard (pipeline plan section 22): reset_opacity clamps ALL
    # opacities to <= 0.01, so a per-event prune floor at or above the clamp makes
    # the whole model prune-eligible after every reset; the budgeted prune then
    # removes 50% of it per densify event until opacities recover (v10: 548k -> 17k
    # in 5 events). Hard error whenever a reset falls inside the densify window.
    if (densify_min_opacity >= 0.01
            and int(opt.opacity_reset_interval) <= int(opt.densify_until_iter)):
        raise RuntimeError(
            "--densify_min_opacity {} >= 0.01, the reset_opacity clamp: every opacity "
            "reset (interval {}) would make the whole model prune-eligible and the "
            "budgeted prune removes 50% per densify event until recovery. Keep it "
            "below 0.01, or move --opacity_reset_interval past --densify_until_iter "
            "if you truly intend this.".format(
                densify_min_opacity, int(opt.opacity_reset_interval)))
    if not (0.0 < final_prune_min_opacity < 1.0):
        raise RuntimeError("--final_prune_min_opacity must be in (0,1), got {}".format(final_prune_min_opacity))
    if not (0.0 < final_prune_score_thresh <= 1.0):
        raise RuntimeError("--final_prune_score_thresh must be in (0,1], got {}".format(final_prune_score_thresh))
    if not math.isfinite(densify_metric_gate) or densify_metric_gate < 0.0:
        raise RuntimeError("--densify_metric_gate must be finite and >= 0, got {}".format(densify_metric_gate))
    # --- Class-scoped densify weighting (FASTGS_ROI_DENSIFY_CLASS_WEIGHTS) ---
    roi_densify_cw_spec = str(getattr(dataset, "roi_densify_class_weights", "") or "")
    roi_densify_cw_groups = None
    if roi_densify_cw_spec:
        if not roi_enabled:
            raise RuntimeError(
                "[ROI-DENSIFY-CW] --roi_densify_class_weights requires --use_roi_masks "
                "(class maps come from the ROI mask products).")
        try:
            roi_densify_cw_groups = parse_densify_class_weights(roi_densify_cw_spec)
        except ValueError as exc:
            raise RuntimeError("[ROI-DENSIFY-CW] bad --roi_densify_class_weights: {}".format(exc))
    # --- Class-scoped scale regularization (FASTGS_ROI_SCALE_REG, scale-reg plan r2):
    # ray-modulated anisotropy hinge on the top-2 log-scales of attributed gaussians.
    scale_reg_spec = str(getattr(dataset, "roi_scale_reg", "") or "")
    scale_reg_groups = None
    scale_reg_r0 = float(getattr(opt, "roi_scale_reg_ratio", 4.0))
    scale_reg_log_r0 = 0.0
    if scale_reg_spec:
        if not roi_enabled:
            raise RuntimeError(
                "[ROI-SCALE-REG] --roi_scale_reg requires --use_roi_masks "
                "(class maps come from the ROI mask products).")
        try:
            scale_reg_groups = parse_scale_reg_spec(scale_reg_spec)
        except ValueError as exc:
            raise RuntimeError("[ROI-SCALE-REG] bad --roi_scale_reg: {}".format(exc))
        if not (math.isfinite(scale_reg_r0) and scale_reg_r0 > 1.0):
            raise RuntimeError(
                "--roi_scale_reg_ratio must be finite and > 1, got {}".format(scale_reg_r0))
        scale_reg_log_r0 = math.log(scale_reg_r0)
        if str(getattr(opt, "optimizer_type", "default")) != "default":
            raise RuntimeError(
                "[ROI-SCALE-REG] requires optimizer_type 'default': sparse_adam steps "
                "only the current view's visible gaussians, silently dropping the "
                "regularizer's off-view _scaling gradients.")
    # --- r4 scale regularization v2 (FASTGS_ROI_SCALE_REG_V2, plan §11.5/§16/§17):
    # preferential-hiddenness hinge + optional plate term, label-only. ---
    scale_reg_v2_spec = str(getattr(dataset, "roi_scale_reg_v2", "") or "")
    scale_reg_v2_on = bool(scale_reg_v2_spec)
    scale_reg_v2_lam = 0.0
    scale_reg_v2_log_r0 = 0.0
    scale_reg_v2_plate_lam = float(getattr(opt, "roi_scale_reg_v2_plate_lambda", 0.0))
    scale_reg_v2_log_rp0 = 0.0
    scale_reg_v2_class = -1
    if scale_reg_v2_on:
        if scale_reg_spec:
            raise RuntimeError(
                "[ROI-SCALE-REG-V2] --roi_scale_reg_v2 and --roi_scale_reg are mutually "
                "exclusive: two scale regularizers stacking on the same _scaling rows is "
                "never an intended configuration.")
        if not roi_enabled:
            raise RuntimeError(
                "[ROI-SCALE-REG-V2] --roi_scale_reg_v2 requires --use_roi_masks "
                "(class maps come from the ROI mask products).")
        try:
            _v2_groups = parse_scale_reg_spec(scale_reg_v2_spec)
        except ValueError as exc:
            raise RuntimeError("[ROI-SCALE-REG-V2] bad --roi_scale_reg_v2: {}".format(exc))
        _v2_lbl = int(getattr(dataset, "roi_label_class_id", 2))
        if not (0 <= _v2_lbl < 5):  # 5-class LightSeg maps
            raise RuntimeError(
                "[ROI-SCALE-REG-V2] --roi_label_class_id {} out of range [0, 4]".format(_v2_lbl))
        if (len(_v2_groups) != 1 or len(_v2_groups[0][1]) != 1
                or int(_v2_groups[0][1][0]) != _v2_lbl):
            raise RuntimeError(
                "[ROI-SCALE-REG-V2] v2 is label-only BY DESIGN (plan §11 correction #2: "
                "union grouping diluted label gradients 3-7x in v13); expected exactly "
                "\"{}:<lambda>\", got {!r}.".format(_v2_lbl, scale_reg_v2_spec))
        scale_reg_v2_lam = float(_v2_groups[0][0])
        scale_reg_v2_class = _v2_lbl
        _v2_r0 = float(getattr(opt, "roi_scale_reg_v2_ratio", 4.0))
        if not (math.isfinite(_v2_r0) and _v2_r0 > 1.0):
            raise RuntimeError(
                "--roi_scale_reg_v2_ratio must be finite and > 1, got {}".format(_v2_r0))
        scale_reg_v2_log_r0 = math.log(_v2_r0)
        if not (math.isfinite(scale_reg_v2_plate_lam) and scale_reg_v2_plate_lam >= 0.0):
            raise RuntimeError(
                "--roi_scale_reg_v2_plate_lambda must be finite and >= 0, got {}".format(
                    scale_reg_v2_plate_lam))
        _v2_rp0 = float(getattr(opt, "roi_scale_reg_v2_plate_ratio", 150.0))
        if not (math.isfinite(_v2_rp0) and _v2_rp0 > 1.0):
            raise RuntimeError(
                "--roi_scale_reg_v2_plate_ratio must be finite and > 1, got {}".format(_v2_rp0))
        scale_reg_v2_log_rp0 = math.log(_v2_rp0)
        if str(getattr(opt, "optimizer_type", "default")) != "default":
            raise RuntimeError(
                "[ROI-SCALE-REG-V2] requires optimizer_type 'default': sparse_adam steps "
                "only the current view's visible gaussians, silently dropping the "
                "regularizer's off-view _scaling gradients.")
    # --- A1 observability/billboard telemetry (FASTGS_ROI_OBS_TELEMETRY): READ-ONLY
    # diagnostics for the r4 calibration; never touches the loss or the model. ---
    roi_obs_telemetry = bool(getattr(dataset, "roi_obs_telemetry", False))
    if roi_obs_telemetry and not roi_enabled:
        raise RuntimeError(
            "[ROI-OBS] --roi_obs_telemetry requires --use_roi_masks "
            "(class maps come from the ROI mask products).")
    if roi_obs_telemetry:
        _obs_lbl = int(getattr(dataset, "roi_label_class_id", 2))
        if not (0 <= _obs_lbl < 5):  # 5-class LightSeg maps; dump indexes column _obs_lbl
            raise RuntimeError(
                "[ROI-OBS] --roi_label_class_id {} out of range [0, 4]".format(_obs_lbl))
    if roi_enabled:
        if opt.roi_norm not in ("roi", "global"):
            raise RuntimeError(
                "[ROI] unknown --roi_norm {!r}: expected 'roi' or 'global'".format(opt.roi_norm))
        if opt.roi_densify_mode not in ("intersect", "blend"):
            raise RuntimeError(
                "[ROI] unknown --roi_densify_mode {!r}: expected 'intersect' or 'blend'".format(
                    opt.roi_densify_mode))
        if dataset.roi_missing not in ("fail_open", "fail_loud"):
            raise RuntimeError(
                "[ROI] unknown --roi_missing {!r}: expected 'fail_open' or 'fail_loud'".format(
                    dataset.roi_missing))
        if getattr(opt, "final_prune_interval", 3000) < 1:
            raise RuntimeError("[ROI] --final_prune_interval must be >= 1")
        warmup = int(getattr(opt, "roi_warmup_iters", 1000))
        ramp = int(getattr(opt, "roi_ramp_iters", 300))
        start = int(getattr(opt, "roi_densify_start_iter", -1))
        roi_densify_start = start if start >= 0 else warmup + ramp
        roi_prune_bg = bool(getattr(opt, "roi_prune_background", False))
        if roi_prune_bg:
            lut = parse_class_weights(dataset.roi_class_weights)
            if float(lut[0]) != 0.0:
                raise RuntimeError(
                    "[ROI] --roi_prune_background requires background weight exactly 0.0 "
                    "(pass 0:0.0 in --roi_class_weights); pruning what a nonzero "
                    "background weight is simultaneously supervising is incoherent."
                )
            # Pruning must never engage while the loss is still unmasked / the
            # densify gate still admits background (prune->regrow churn); clamp.
            if opt.roi_prune_start_iter < roi_densify_start:
                print("[ROI] clamping roi_prune_start_iter {} -> {} (must not precede "
                      "the densify gate / end of warmup+ramp)".format(
                          opt.roi_prune_start_iter, roi_densify_start))
                opt.roi_prune_start_iter = roi_densify_start
    # ---------------------------------------------------------------------------

    scene = Scene(dataset, gaussians, roi_for_training=roi_enabled)
    gaussians.training_setup(opt)
    if checkpoint:
        # weights_only=False: torch >= 2.6 flipped the default to True, which
        # rejects the (capture(), iteration) tuple; explicit False restores the
        # historical behavior on every torch version >= 1.13.
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)
        if scale_reg_groups is not None:
            print("[ROI-SCALE-REG] WARNING: resuming from a checkpoint; the regularizer "
                  "is inactive until the first densify/final-prune event after resume "
                  "re-derives the attribution.")
        if scale_reg_v2_on:
            print("[ROI-SCALE-REG-V2] WARNING: resuming from a checkpoint; the regularizer "
                  "is inactive until the first densify/final-prune event after resume "
                  "re-derives the observability cache.")
        if roi_prune_bg:
            print("[ROI] WARNING: resuming from a checkpoint; roi_zero_rounds background-"
                  "prune counters are not checkpointed and restart from zero — background "
                  "pruning is delayed by up to roi_prune_min_rounds score events, never "
                  "wrongly applied.")

    # --- ROI mask coverage check / banner (after Scene: needs loaded cameras) ---
    if roi_enabled:
        train_cams = scene.getTrainCameras()
        roi_failopen_count = sum(
            1 for cam in train_cams if getattr(cam, "roi_weight", None) is None
        )
        if roi_refine:
            n_lb = sum(1 for cam in train_cams
                       if getattr(cam, "roi_label_bin", None) is not None)
            n_masked = len(train_cams) - roi_failopen_count
            if n_lb < n_masked:
                raise RuntimeError(
                    "[ROI-REFINE] only {}/{} masked training views carry a label stencil; "
                    "--roi_keep_label_bin should have populated all of them.".format(
                        n_lb, n_masked))
            print("[ROI-REFINE] enabled | start iter {} | label mult {} | ramp {} | "
                  "stencils {}/{} views".format(
                      roi_refine_start, roi_refine_mult, roi_refine_ramp, n_lb,
                      len(train_cams)))
        if roi_vieww_json:
            n_scaled = 0
            with torch.no_grad():
                for cam in train_cams:
                    w = getattr(cam, "roi_weight", None)
                    if w is not None and float(w.float().max().item()) > 1.0:
                        n_scaled += 1
            print("[ROI-VIEWW] enabled | weights json {} | views with max weight > 1.0: "
                  "{}/{}".format(roi_vieww_json, n_scaled, len(train_cams)))
        if roi_densify_cw_groups is not None:
            n_cm = sum(1 for cam in train_cams
                       if getattr(cam, "roi_class_map", None) is not None)
            n_masked = len(train_cams) - roi_failopen_count
            if n_cm < n_masked:
                raise RuntimeError(
                    "[ROI-DENSIFY-CW] only {}/{} masked training views carry a class map; "
                    "--roi_densify_class_weights should have populated all of them.".format(
                        n_cm, n_masked))
            print("[ROI-DENSIFY-CW] enabled | groups {} | class maps {}/{} views".format(
                [(w, ids) for w, ids in roi_densify_cw_groups], n_cm, len(train_cams)))
        if scale_reg_groups is not None:
            n_cm = sum(1 for cam in train_cams
                       if getattr(cam, "roi_class_map", None) is not None)
            n_masked = len(train_cams) - roi_failopen_count
            if n_cm < n_masked:
                raise RuntimeError(
                    "[ROI-SCALE-REG] only {}/{} masked training views carry a class map; "
                    "--roi_scale_reg should have populated all of them.".format(
                        n_cm, n_masked))
            print("[ROI-SCALE-REG] enabled | groups {} | r0 {} | class maps {}/{} views".format(
                [(l, ids) for l, ids in scale_reg_groups], scale_reg_r0, n_cm,
                len(train_cams)))
        if scale_reg_v2_on:
            n_cm = sum(1 for cam in train_cams
                       if getattr(cam, "roi_class_map", None) is not None)
            n_masked = len(train_cams) - roi_failopen_count
            if n_masked <= 0 or n_cm < n_masked:
                raise RuntimeError(
                    "[ROI-SCALE-REG-V2] {}/{} masked training views carry a class map "
                    "(zero masked views = broken mask export); --roi_scale_reg_v2 needs "
                    "class maps on every masked view.".format(n_cm, n_masked))
            print("[ROI-SCALE-REG-V2] enabled | label class {} lambda {} | r0 {} | plate "
                  "lambda {} ratio {} | class maps {}/{} views".format(
                      scale_reg_v2_class, scale_reg_v2_lam, math.exp(scale_reg_v2_log_r0),
                      scale_reg_v2_plate_lam, math.exp(scale_reg_v2_log_rp0),
                      n_cm, len(train_cams)))
        if roi_obs_telemetry:
            n_cm = sum(1 for cam in train_cams
                       if getattr(cam, "roi_class_map", None) is not None)
            n_masked = len(train_cams) - roi_failopen_count
            if n_masked <= 0 or n_cm < n_masked:
                raise RuntimeError(
                    "[ROI-OBS] {}/{} masked training views carry a class map "
                    "(zero masked views = broken mask export); --roi_obs_telemetry "
                    "needs class maps on every masked view.".format(n_cm, n_masked))
            print("[ROI-OBS] enabled | label class {} | class maps {}/{} views | "
                  "READ-ONLY telemetry (no loss change)".format(
                      int(getattr(dataset, "roi_label_class_id", 2)), n_cm, len(train_cams)))
        max_frac = float(getattr(dataset, "roi_max_failopen_frac", 0.10))
        if len(train_cams) > 0 and roi_failopen_count / len(train_cams) > max_frac:
            raise RuntimeError(
                "[ROI] {}/{} training views have no usable mask (fail-open), exceeding "
                "--roi_max_failopen_frac={}. Mass mask failure means the export step is "
                "broken; refusing to train near-unmasked under an ROI method hash.".format(
                    roi_failopen_count, len(train_cams), max_frac
                )
            )
        with torch.no_grad():
            fg = [
                float((cam.roi_bin > 0).float().mean().item())
                for cam in train_cams
                if getattr(cam, "roi_bin", None) is not None
            ]
        if fg:
            roi_mean_fg_frac = sum(fg) / len(fg)
            print(
                "[ROI] enabled | norm={} | mean fg frac={:.4f} (implied roi-norm grad "
                "scale ~{:.2f}x) | fail-open views={}/{} | densify gate from iter {} | "
                "background prune={}".format(
                    opt.roi_norm,
                    roi_mean_fg_frac,
                    1.0 / max(roi_mean_fg_frac, 1e-6),
                    roi_failopen_count,
                    len(train_cams),
                    roi_densify_start,
                    roi_prune_bg,
                )
            )
    # ---------------------------------------------------------------------------

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    # record time
    optim_start = torch.cuda.Event(enable_timing=True)
    optim_end = torch.cuda.Event(enable_timing=True)
    total_time = 0.0

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    # train_stats.json collection
    stats_iter_samples = []
    stats_densify_events = []
    stats_bg_pruned_total = 0
    roi_starvation_warned = False
    # Scale-reg state: attribution + direction resultant are valid only for the
    # population they were computed on; refreshed after every mutation (see the
    # two attribute_gaussians_by_class call sites), None before the first refresh.
    scale_reg_attr = None
    scale_reg_rbar = None
    scale_reg_refresh_i = 0
    stats_scale_reg_events = []
    scale_reg_v2_h = None
    scale_reg_v2_elig1 = None
    scale_reg_v2_elig2 = None
    scale_reg_v2_refresh_i = 0
    stats_scale_reg_v2_events = []
    # A1 telemetry heartbeat state (READ-ONLY; every 10th densify event)
    obs_event_i = 0
    stats_obs_events = []

    for iteration in range(first_iter, opt.iterations + 1):

        if websockets:
            if network_gui_ws.curr_id >= 0 and network_gui_ws.curr_id < len(scene.getTrainCameras()):
                cam = scene.getTrainCameras()[network_gui_ws.curr_id]
                net_image = render_fastgs(cam, gaussians, pipe, background, opt.mult, 1.0)["render"]
                network_gui_ws.latest_width = cam.image_width
                network_gui_ws.latest_height = cam.image_height
                network_gui_ws.latest_result = net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

        iter_start.record()
        
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        _ = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, bg, opt.mult)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        roi_W = None
        if roi_enabled and iteration > opt.roi_warmup_iters:
            roi_W = getattr(viewpoint_cam, "roi_weight", None)
        if roi_W is not None:
            roi_W = roi_W.cuda().float()
            if opt.roi_ramp_iters > 0:
                ramp_t = min(1.0, (iteration - opt.roi_warmup_iters) / float(opt.roi_ramp_iters))
            else:
                ramp_t = 1.0
            if ramp_t < 1.0:
                # Linear fade-in: a hard normalization switch shocks Adam's second
                # moments right as densify events compare raw gradients to fixed
                # thresholds; ramping removes the transient.
                roi_W = (1.0 - ramp_t) + ramp_t * roi_W
            # Late label refinement: boost label pixels only, only after the
            # densification window (roi_refine_start >= densify_until_iter, clamped
            # above), with its own linear ramp. Multiplies AFTER the warmup ramp;
            # ramp_t is 1.0 by then so composition order is moot but kept explicit.
            if roi_refine and iteration > roi_refine_start:
                label_bin = getattr(viewpoint_cam, "roi_label_bin", None)
                if label_bin is not None:
                    if roi_refine_ramp > 0:
                        refine_t = min(1.0, (iteration - roi_refine_start) / float(roi_refine_ramp))
                    else:
                        refine_t = 1.0
                    eff_mult = 1.0 + (roi_refine_mult - 1.0) * refine_t
                    roi_W = roi_W * (1.0 + (eff_mult - 1.0) * label_bin.cuda().float())
            Ll1 = masked_l1(image, gt_image, roi_W, opt.roi_norm)
            ssim_value = masked_ssim(image.unsqueeze(0), gt_image.unsqueeze(0), roi_W, opt.roi_norm)
        else:
            Ll1 = l1_loss(image, gt_image)
            ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        # Ray-modulated scale regularization (scale-reg plan r2): for attributed
        # gaussians, hinge on ls1 - ls2 + ln|a1 . r_bar| - ln r0. Gradient reaches
        # _scaling only; axis and direction resultant are frozen evidence. Inactive
        # until the first attribution refresh (~first densify event).
        if scale_reg_groups is not None and scale_reg_attr is not None:
            n_now = gaussians.get_xyz.shape[0]
            if scale_reg_attr.shape[0] != n_now:
                raise RuntimeError(
                    "[ROI-SCALE-REG] attribution tensor is stale ({} rows vs {} "
                    "gaussians): a population mutation was not followed by a "
                    "refresh.".format(scale_reg_attr.shape[0], n_now))
            sub = torch.nonzero(scale_reg_attr >= 0, as_tuple=True)[0]
            if sub.numel() > 0:
                ls_sub = gaussians._scaling[sub]
                top2 = torch.topk(ls_sub, 2, dim=1)
                with torch.no_grad():
                    rot_sub = build_rotation(gaussians._rotation[sub])
                    # Top-1 index from the SAME topk that provides the hinge values,
                    # so a1 always matches the axis the hinge calls "longest" (an
                    # exact tie would otherwise let argmax and topk disagree).
                    amax = top2.indices[:, 0]
                    a1 = rot_sub[torch.arange(sub.numel(), device=sub.device), :, amax]
                    align = (a1 * scale_reg_rbar[sub]).sum(dim=1).abs().clamp_min(1e-3)
                pen = torch.relu(top2.values[:, 0] - top2.values[:, 1]
                                 + torch.log(align) - scale_reg_log_r0)
                attr_sub = scale_reg_attr[sub]
                for gi, (lam, _ids) in enumerate(scale_reg_groups):
                    gsel = attr_sub == gi
                    if bool(gsel.any()):
                        loss = loss + float(lam) * pen[gsel].mean()
        # r4 scale reg v2 (plan §11.5/§17): preferential-hiddenness hinge (+ plate
        # term when armed) over the refresh-cached eligibility. Gradients reach
        # _scaling only; h and eligibility are frozen evidence between refreshes.
        # Inactive until the first observability refresh (~first densify event).
        # Row-alignment note: reset_opacity is population-size-NEUTRAL
        # (replace_tensor_to_optimizer preserves rows), so only the densify and
        # final-prune blocks can invalidate this cache — both refresh it.
        if scale_reg_v2_on and scale_reg_v2_h is not None:
            n_now = gaussians.get_xyz.shape[0]
            if scale_reg_v2_h.shape[0] != n_now:
                raise RuntimeError(
                    "[ROI-SCALE-REG-V2] observability cache is stale ({} rows vs {} "
                    "gaussians): a population mutation was not followed by a "
                    "refresh.".format(scale_reg_v2_h.shape[0], n_now))
            v2 = scale_reg_v2_penalty(
                gaussians._scaling, scale_reg_v2_h, scale_reg_v2_elig1,
                scale_reg_v2_elig2, scale_reg_v2_log_r0, scale_reg_v2_log_rp0)
            loss = loss + scale_reg_v2_lam * v2["term1"]
            if scale_reg_v2_plate_lam > 0.0:
                loss = loss + scale_reg_v2_plate_lam * v2["term2"]
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            iter_time = iter_start.elapsed_time(iter_end)
            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_time, testing_iterations, scene, render_fastgs, (pipe, background, opt.mult))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            optim_start.record()
            
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    my_viewpoint_stack = scene.getTrainCameras().copy()
                    camlist = sampling_cameras(my_viewpoint_stack, getattr(opt, "score_num_cameras", 10))

                    roi_cfg = None
                    if roi_enabled:
                        roi_cfg = SimpleNamespace(
                            densify_active=(iteration >= roi_densify_start),
                            densify_mode=opt.roi_densify_mode,
                            bg_scale=opt.roi_densify_bg_scale,
                            track_roi_touch=roi_prune_bg,
                            last_flagged_px_mean=None,
                            class_weight_groups=roi_densify_cw_groups,
                        )

                    # The multiview consistent densification of fastgs
                    importance_score, pruning_score, roi_touch = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt, DENSIFY=True, roi_cfg=roi_cfg)
                    if roi_cfg is not None and roi_cfg.track_roi_touch:
                        # Counter update must precede the resize below (counts are
                        # aligned to the current population; the counter itself is
                        # remapped through prune/cat inside the model).
                        gaussians.update_roi_zero_rounds(roi_touch)

                    n_before_densify = gaussians.get_xyz.shape[0]
                    gaussians.densify_and_prune_fastgs(max_screen_size = size_threshold,
                                                min_opacity = densify_min_opacity,
                                                extent = scene.cameras_extent,
                                                radii=radii,
                                                args = opt,
                                                importance_score = importance_score,
                                                pruning_score = pruning_score)

                    if roi_enabled and roi_prune_bg and iteration >= opt.roi_prune_start_iter:
                        pruned_bg = gaussians.prune_background_by_rounds(
                            opt.roi_prune_min_rounds, opt.roi_prune_min_keep)
                        if pruned_bg:
                            stats_bg_pruned_total += pruned_bg
                            print(f"[ITER {iteration}] ROI background prune: removed {pruned_bg}")

                    # Scale-reg attribution refresh: MUST follow the last population
                    # mutation in this block (densify_and_prune + bg-rounds prune
                    # above); the cached tensors are row-aligned until the next one.
                    if scale_reg_groups is not None:
                        scale_reg_attr, scale_reg_rbar = attribute_gaussians_by_class(
                            camlist, gaussians, pipe, bg, opt, scale_reg_groups)
                        scale_reg_refresh_i += 1
                        if scale_reg_refresh_i % 10 == 1:
                            valid = scale_reg_attr >= 0
                            counts = [int((scale_reg_attr == gi).sum().item())
                                      for gi in range(len(scale_reg_groups))]
                            if bool(valid.any()):
                                rho_med = float(scale_reg_rbar[valid].norm(dim=1).median().item())
                                t2 = torch.topk(gaussians._scaling[valid].detach(), 2, dim=1).values
                                rr = t2[:, 0] - t2[:, 1]
                                rr_mean, rr_p95 = float(rr.mean().item()), float(torch.quantile(rr, 0.95).item())
                            else:
                                rho_med, rr_mean, rr_p95 = -1.0, -1.0, -1.0
                            print("[ROI-SCALE-REG] iter {} refresh {}: attributed {} | rho med "
                                  "{:.3f} | ratio_log mean {:.3f} p95 {:.3f}".format(
                                      iteration, scale_reg_refresh_i, counts, rho_med, rr_mean, rr_p95))
                            stats_scale_reg_events.append(
                                {"iter": iteration, "attributed": counts, "rho_median": rho_med,
                                 "ratio_log_mean": rr_mean, "ratio_log_p95": rr_p95})

                    # v2 observability refresh: MUST follow the last population
                    # mutation in this block (same row-alignment contract as the
                    # v1 refresh above). Heartbeat-mode telemetry (2 stencil
                    # renders/cam) supplies h + pool; the billboard kNN runs only
                    # when the plate term is armed.
                    if scale_reg_v2_on:
                        _t = observability_telemetry(
                            camlist, gaussians, pipe, bg, opt, scale_reg_v2_class,
                            with_billboard=(scale_reg_v2_plate_lam > 0.0))
                        scale_reg_v2_h = _t["h"].detach()
                        _pool = _t["pool_mask"] & (_t["support"] >= 2.0)
                        scale_reg_v2_elig1 = _pool
                        if scale_reg_v2_plate_lam > 0.0:
                            # plate gate frozen from the §17 npz calibration:
                            # billboard (|a3.n_local| < 0.5) & h > 0.3
                            _bb = _t["billboard"]
                            scale_reg_v2_elig2 = (_pool & (scale_reg_v2_h > 0.3)
                                                  & torch.isfinite(_bb) & (_bb.abs() < 0.5))
                        else:
                            scale_reg_v2_elig2 = torch.zeros_like(_pool)
                        scale_reg_v2_refresh_i += 1
                        if scale_reg_v2_refresh_i % 10 == 1:
                            with torch.no_grad():
                                _p = scale_reg_v2_penalty(
                                    gaussians._scaling.detach(), scale_reg_v2_h,
                                    scale_reg_v2_elig1, scale_reg_v2_elig2,
                                    scale_reg_v2_log_r0, scale_reg_v2_log_rp0,
                                    with_stats=True)
                                _hh = scale_reg_v2_h[scale_reg_v2_elig1]
                                _hmed = float(_hh.median().item()) if _hh.numel() else -1.0
                            print("[ROI-SCALE-REG-V2] iter {} refresh {}: elig {} | h med "
                                  "{:.3f} | act1 {:.3f} pen1 {:.5f} | elig2 {} act2 {:.3f} "
                                  "pen2 {:.5f}".format(
                                      iteration, scale_reg_v2_refresh_i, _p["n1"], _hmed,
                                      _p["act1"], float(_p["term1"]), _p["n2"], _p["act2"],
                                      float(_p["term2"])))
                            stats_scale_reg_v2_events.append(
                                {"iter": iteration, "eligible": _p["n1"], "h_median": _hmed,
                                 "active_frac_t1": _p["act1"], "pen_t1": float(_p["term1"]),
                                 "eligible_t2": _p["n2"], "active_frac_t2": _p["act2"],
                                 "pen_t2": float(_p["term2"])})

                    # A1 telemetry heartbeat: cheap 2-render/cam probe every 10th
                    # densify event; read-only, after all mutators (row-aligned).
                    if roi_obs_telemetry:
                        obs_event_i += 1
                        if obs_event_i % 10 == 1:
                            # Fail-soft (review finding): a read-only diagnostic
                            # must never kill a training run — same discipline as
                            # the final dump.
                            try:
                                tel = observability_telemetry(
                                    camlist, gaussians, pipe, bg, opt,
                                    int(getattr(dataset, "roi_label_class_id", 2)))
                                pool = tel["pool_mask"]
                                n_pool = int(pool.sum().item())
                                bb = tel["billboard"][pool]
                                bb = bb[torch.isfinite(bb)]
                                if n_pool:
                                    # billboard gates separately: it is NaN for
                                    # pools <= K_LOCAL even when h/purity are fine
                                    bb_frac = (float((bb < 0.5).float().mean().item())
                                               if bb.numel() else -1.0)
                                    h_med = float(tel["h"][pool].median().item())
                                    h_p95 = float(torch.quantile(tel["h"][pool], 0.95).item())
                                    sup_med = float(tel["support"][pool].median().item())
                                    pur_med = float(tel["purity"][pool].median().item())
                                else:
                                    bb_frac = h_med = h_p95 = sup_med = pur_med = -1.0
                                print("[ROI-OBS] iter {} pool {} | purity med {:.2f} | "
                                      "support med {:.1f} | h med {:.3f} p95 {:.3f} | "
                                      "billboard frac {:.3f}".format(
                                          iteration, n_pool, pur_med, sup_med, h_med,
                                          h_p95, bb_frac))
                                stats_obs_events.append({
                                    "iter": iteration, "pool": n_pool,
                                    "purity_med": pur_med, "support_med": sup_med,
                                    "h_med": h_med, "h_p95": h_p95,
                                    "billboard_frac_lt05": bb_frac})
                            except Exception as e:
                                print("[ROI-OBS] WARNING: heartbeat failed at iter {}: {}".format(
                                    iteration, e))

                    flagged_mean = roi_cfg.last_flagged_px_mean if roi_cfg is not None else None
                    stats_densify_events.append({
                        "iter": iteration,
                        "n_before": int(n_before_densify),
                        "n_after": int(gaussians.get_xyz.shape[0]),
                        "roi_flagged_px_mean": flagged_mean,
                    })
                    if (roi_cfg is not None and roi_cfg.densify_active
                            and flagged_mean is not None and flagged_mean < 10.0
                            and not roi_starvation_warned):
                        print(f"[ITER {iteration}] WARNING: ROI densify map has ~no flagged "
                              f"pixels (mean {flagged_mean:.1f}/view) - ROI densification is "
                              f"starving; check loss_thresh / mask quality.")
                        roi_starvation_warned = True

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # The multiview consistent pruning of fastgs. We do it every 3k iterations after 15k
            # In this stage, the model converge basically. So we can prune more aggressively without degrading rendering quality.
            # You can check the rendering results of 20K iterations in arxiv version (https://arxiv.org/abs/2511.04283), the rendering quality is already very good.
            if (getattr(opt, "final_prune_interval", 3000) > 0
                    and iteration % opt.final_prune_interval == 0
                    and iteration > opt.densify_until_iter
                    and iteration < getattr(opt, "final_prune_until_iter", 30_000)):
                my_viewpoint_stack = scene.getTrainCameras().copy()
                camlist = sampling_cameras(my_viewpoint_stack, getattr(opt, "score_num_cameras", 10))

                roi_cfg = None
                if roi_enabled and roi_prune_bg:
                    roi_cfg = SimpleNamespace(
                        densify_active=False,
                        densify_mode=opt.roi_densify_mode,
                        bg_scale=opt.roi_densify_bg_scale,
                        track_roi_touch=True,
                        last_flagged_px_mean=None,
                    )

                _, pruning_score, roi_touch = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt, roi_cfg=roi_cfg)
                if roi_cfg is not None:
                    gaussians.update_roi_zero_rounds(roi_touch)

                before = gaussians.get_xyz.shape[0]

                # Gentler late prune for label-mode stability
                gaussians.final_prune_fastgs(
                    min_opacity=final_prune_min_opacity,
                    pruning_score=pruning_score,
                    score_thresh=final_prune_score_thresh,
                    min_keep=1024,
                )

                after = gaussians.get_xyz.shape[0]
                print(f"[ITER {iteration}] final prune: {before} -> {after}")

                if roi_enabled and roi_prune_bg and iteration >= opt.roi_prune_start_iter:
                    pruned_bg = gaussians.prune_background_by_rounds(
                        opt.roi_prune_min_rounds, opt.roi_prune_min_keep)
                    if pruned_bg:
                        stats_bg_pruned_total += pruned_bg
                        print(f"[ITER {iteration}] ROI background prune: removed {pruned_bg}")

                # Scale-reg attribution refresh: follows final_prune + bg-rounds
                # prune, the last mutators in this block (same invariant as the
                # densify-block refresh above).
                if scale_reg_groups is not None:
                    scale_reg_attr, scale_reg_rbar = attribute_gaussians_by_class(
                        camlist, gaussians, pipe, bg, opt, scale_reg_groups)
                    scale_reg_refresh_i += 1
                if scale_reg_v2_on:
                    _t = observability_telemetry(
                        camlist, gaussians, pipe, bg, opt, scale_reg_v2_class,
                        with_billboard=(scale_reg_v2_plate_lam > 0.0))
                    scale_reg_v2_h = _t["h"].detach()
                    _pool = _t["pool_mask"] & (_t["support"] >= 2.0)
                    scale_reg_v2_elig1 = _pool
                    if scale_reg_v2_plate_lam > 0.0:
                        _bb = _t["billboard"]
                        scale_reg_v2_elig2 = (_pool & (scale_reg_v2_h > 0.3)
                                              & torch.isfinite(_bb) & (_bb.abs() < 0.5))
                    else:
                        scale_reg_v2_elig2 = torch.zeros_like(_pool)
                    scale_reg_v2_refresh_i += 1
        
            # Optimization step
            if iteration < opt.iterations:
                if opt.optimizer_type == "default":
                    gaussians.optimizer_step(iteration)
                elif opt.optimizer_type == "sparse_adam":
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)

            # record time
            optim_end.record()
            torch.cuda.synchronize()
            optim_time = optim_start.elapsed_time(optim_end)
            total_time += (iter_time + optim_time) / 1e3

            if iteration % 500 == 0:
                stats_iter_samples.append({
                    "iter": iteration,
                    "ema_loss": ema_loss_for_log,
                    "iter_time_ms": iter_time,
                    "gaussian_count": int(gaussians.get_xyz.shape[0]),
                })

            # Preemption checkpoint save (restores the stock-3DGS block this fork
            # dropped; requeue-resume plan 2026-08-28). Atomic tmp + os.replace so
            # a mid-save preemption never leaves a truncated checkpoint (same
            # pattern as the obs_telemetry npz dump); only the newest survives so
            # the shared-FS footprint stays at one checkpoint per run.
            if iteration in checkpoint_iterations:
                ckpt_root = checkpoint_dir if checkpoint_dir else dataset.model_path
                os.makedirs(ckpt_root, exist_ok=True)
                ckpt_path = os.path.join(ckpt_root, "chkpnt{}.pth".format(iteration))
                print("\n[ITER {}] Saving Checkpoint to {}".format(iteration, ckpt_path))
                # This fork's capture() takes optimizer_type (stock 3DGS takes
                # none); 'default' is enforced by the startup guard above so the
                # tuple always matches what restore() unpacks.
                torch.save((gaussians.capture(opt.optimizer_type), iteration), ckpt_path + ".tmp")
                os.replace(ckpt_path + ".tmp", ckpt_path)
                for fname in os.listdir(ckpt_root):
                    if (fname.startswith("chkpnt") and fname.endswith(".pth")
                            and fname != os.path.basename(ckpt_path)):
                        try:
                            older_iter = int(fname[len("chkpnt"):-len(".pth")])
                        except ValueError:
                            continue
                        if older_iter < iteration:
                            try:
                                os.remove(os.path.join(ckpt_root, fname))
                            except OSError:
                                pass

    if opt.iterations in saving_iterations:
        # Re-save the terminal PLY: the in-loop save runs BEFORE the same
        # iteration's densify/late-prune blocks, so a run ending on a prune
        # event would otherwise persist a pre-prune point cloud. Overwriting
        # here is a no-op when no terminal event fired.
        print("\n[ITER {}] Saving Gaussians (post-prune terminal save)".format(opt.iterations))
        scene.save(opt.iterations)

    print(f"Gaussian number: {gaussians._xyz.shape[0]}")
    print(f"Training time: {total_time}")

    stats = {
        "wall_clock_s": total_time,
        "iterations": opt.iterations,
        "final_gaussian_count": int(gaussians._xyz.shape[0]),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "iter_samples_by_500": stats_iter_samples,
        "densify_events": stats_densify_events,
        "roi": {
            "enabled": roi_enabled,
            "norm": getattr(opt, "roi_norm", None) if roi_enabled else None,
            "densify_mode": getattr(opt, "roi_densify_mode", None) if roi_enabled else None,
            "prune_background": roi_prune_bg,
            "mean_fg_frac": roi_mean_fg_frac,
            "roi_engaged_iter": (opt.roi_warmup_iters + 1) if roi_enabled else None,
            "densify_gate_start_iter": roi_densify_start if roi_enabled else None,
            "failopen_views": roi_failopen_count if roi_enabled else 0,
            "pruned_background_total": stats_bg_pruned_total,
        },
    }
    # Arm keys are added only when the respective arm ran, so arm-off runs keep a
    # byte-identical train_stats.json (same discipline as the toolchain JSON).
    if roi_vieww_json:
        stats["roi"]["view_weights_json"] = roi_vieww_json
    if roi_refine:
        stats["roi"]["late_refine"] = {
            "enabled": True,
            "start_iter": roi_refine_start,
            "label_mult": roi_refine_mult,
            "ramp_iters": roi_refine_ramp,
        }
    if roi_densify_cw_spec:
        stats["roi"]["densify_class_weights"] = roi_densify_cw_spec
    # Solidity dials recorded only when moved off their historical literals, so
    # default runs keep a byte-identical train_stats.json.
    _dials = {"densify_min_opacity": (densify_min_opacity, 0.005),
              "final_prune_min_opacity": (final_prune_min_opacity, 0.05),
              "final_prune_score_thresh": (final_prune_score_thresh, 0.95),
              "densify_metric_gate": (densify_metric_gate, 5.0)}
    _moved = {k: v for k, (v, d) in _dials.items() if v != d}
    if _moved:
        stats["solidity_dials"] = _moved
    if scale_reg_groups is not None:
        stats["scale_reg"] = {
            "spec": scale_reg_spec,
            "r0": scale_reg_r0,
            "groups": [[lam, ids] for lam, ids in scale_reg_groups],
            "refreshes": scale_reg_refresh_i,
            "events": stats_scale_reg_events,
        }
    if scale_reg_v2_on:
        stats["scale_reg_v2"] = {
            "spec": scale_reg_v2_spec,
            "r0": math.exp(scale_reg_v2_log_r0),
            "plate_lambda": scale_reg_v2_plate_lam,
            "plate_ratio": math.exp(scale_reg_v2_log_rp0),
            "refreshes": scale_reg_v2_refresh_i,
            "events": stats_scale_reg_v2_events,
        }
    if roi_obs_telemetry:
        # Final per-gaussian dump over ALL masked cameras with full per-class
        # counts (classes partition the map, so purity AND margin are exact).
        # Fail-soft: end-of-run diagnostics must never kill a finished train.
        stats["obs_telemetry"] = {"npz": None, "cams": 0, "events": stats_obs_events}
        try:
            dump_cams = [c for c in scene.getTrainCameras()
                         if getattr(c, "roi_class_map", None) is not None]
            tel = observability_telemetry(
                dump_cams, gaussians, pipe, bg, opt,
                int(getattr(dataset, "roi_label_class_id", 2)), per_class=True)
            arrs = dict(
                xyz=gaussians.get_xyz.detach().cpu().numpy().astype(np.float32),
                scales_log=gaussians._scaling.detach().cpu().numpy().astype(np.float32),
                opacity_logit=gaussians._opacity.detach().reshape(-1).cpu().numpy().astype(np.float32),
                counts=tel["counts"].cpu().numpy().astype(np.float32),
                purity=tel["purity"].cpu().numpy().astype(np.float32),
                margin=tel["margin"].cpu().numpy().astype(np.float32),
                support=tel["support"].cpu().numpy().astype(np.float32),
                rho=tel["rho"].cpu().numpy().astype(np.float32),
                c1=tel["c1"].cpu().numpy().astype(np.float32),
                c2=tel["c2"].cpu().numpy().astype(np.float32),
                h=tel["h"].cpu().numpy().astype(np.float32),
                billboard=tel["billboard"].cpu().numpy().astype(np.float32),
                pool_mask=tel["pool_mask"].cpu().numpy(),
            )
            npz_path = os.path.join(dataset.model_path, "obs_telemetry.npz")
            tmp_path = os.path.join(dataset.model_path, "obs_telemetry_tmp.npz")
            np.savez_compressed(tmp_path, **arrs)
            os.replace(tmp_path, npz_path)  # atomic: no truncated npz on preemption
            stats["obs_telemetry"].update(npz="obs_telemetry.npz", cams=len(dump_cams))
            print("[ROI-OBS] telemetry dump written to {} ({} cams)".format(
                npz_path, len(dump_cams)))
        except Exception as e:
            print("[ROI-OBS] WARNING: telemetry dump failed: {}".format(e))
    try:
        stats_path = os.path.join(dataset.model_path, "train_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Training stats written to {stats_path}")
    except OSError as e:
        print(f"WARNING: could not write train_stats.json: {e}")
    
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    # Default [] (stock-3DGS convention): before the save block was restored this
    # default was dead code; [30_000] would now write a checkpoint at the final
    # iteration of a bare default run, silently changing default behavior.
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # Where checkpoints are written (default: the model dir). The pipeline passes
    # a shared-FS dir here so a requeued job on a different node can resume;
    # node-local model dirs die with the node.
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--websockets", action='store_true', default=False)
    parser.add_argument("--benchmark_dir", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if(args.websockets):
        network_gui_ws.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.websockets,
        checkpoint_dir=args.checkpoint_dir
    )

    # All done
    print("\nTraining complete.")
