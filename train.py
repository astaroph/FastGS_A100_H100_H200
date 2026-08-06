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
import os, random, time, json
from random import randint
from types import SimpleNamespace
from lpipsPyTorch import lpips
from utils.loss_utils import l1_loss
from utils.roi_utils import parse_class_weights, masked_l1, masked_ssim
from fused_ssim import fused_ssim as fast_ssim
from gaussian_renderer import render_fastgs, network_gui_ws
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
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

from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, websockets):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    roi_enabled = bool(getattr(dataset, "use_roi_masks", False))

    # --- ROI config validation (BEFORE Scene: loadCam consumes roi_missing during
    # scene construction, and a typo'd enum must fail before any GPU-heavy work) --
    roi_densify_start = 0
    roi_prune_bg = False
    roi_failopen_count = 0
    roi_mean_fg_frac = None
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
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # --- ROI mask coverage check / banner (after Scene: needs loaded cameras) ---
    if roi_enabled:
        train_cams = scene.getTrainCameras()
        roi_failopen_count = sum(
            1 for cam in train_cams if getattr(cam, "roi_weight", None) is None
        )
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
            Ll1 = masked_l1(image, gt_image, roi_W, opt.roi_norm)
            ssim_value = masked_ssim(image.unsqueeze(0), gt_image.unsqueeze(0), roi_W, opt.roi_norm)
        else:
            Ll1 = l1_loss(image, gt_image)
            ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
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
                                                min_opacity = 0.005,
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
                    min_opacity=0.05,
                    pruning_score=pruning_score,
                    score_thresh=0.95,
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
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
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
        args.websockets
    )

    # All done
    print("\nTraining complete.")
