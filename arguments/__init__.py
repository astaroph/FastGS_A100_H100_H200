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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False

        # ROI (FastGS + LightSeg) parameters â€” all default-off / byte-identical when
        # use_roi_masks is False. See docs/FASTGS_ROI_lightseg_implementation_plan_2026-08-06.md.
        self.use_roi_masks = False
        self.roi_masks_dir = ""
        self.roi_class_weights = "0:0.15,1:1.0,2:1.0,3:1.0,4:1.0"
        self.roi_dilate_px = 12
        self.roi_missing = "fail_open"  # or "fail_loud"
        self.roi_max_failopen_frac = 0.10
        # Per-view clarity scalars (FASTGS_ROI_VIEW_WEIGHTING): path to the
        # roi_view_weights.json emitted by the pipeline's stats tool. "" = off.
        self.roi_view_weights_json = ""
        # Late label refinement (FASTGS_ROI_LATE_LABEL_REFINE): keep a per-camera
        # dilated label stencil at scene load. Class id defaults to the LightSeg
        # 5-class map's label id; the view-weights JSON overrides it when present.
        self.roi_keep_label_bin = False
        self.roi_label_class_id = 2
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.separate_sh = True
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025 
        self.shfeature_lr = 0.005 
        self.opacity_lr = 0.025 
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.001
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        
        # fastgs parameters
        self.loss_thresh = 0.1
        self.grad_abs_thresh = 0.0012  
        self.highfeature_lr = 0.005
        self.lowfeature_lr = 0.0025
        self.grad_thresh = 0.0002
        self.dense = 0.001
        self.mult = 0.5      # multiplier for the compact box to control the tile number of each splat

        # ROI (FastGS + LightSeg) parameters â€” consumed by train.py's ROI training-loop
        # integration (loss weighting, densify gating, background prune, schedules).
        # NOTE: OptimizationParams are NOT persisted in cfg_args (only ModelParams are).
        # See docs/FASTGS_ROI_lightseg_implementation_plan_2026-08-06.md Â§4.1.
        self.roi_warmup_iters = 1000
        self.roi_ramp_iters = 300
        self.roi_norm = "global"  # or "roi"
        # -1 means roi_warmup_iters + roi_ramp_iters (resolved at consumption time, not here).
        self.roi_densify_start_iter = -1
        self.roi_densify_mode = "intersect"  # or "blend"
        self.roi_densify_bg_scale = 0.10
        self.roi_prune_background = False
        self.roi_prune_start_iter = 3000
        self.roi_prune_min_rounds = 2
        self.roi_prune_min_keep = 1024
        # Late label refinement (FASTGS_ROI_LATE_LABEL_REFINE): boost label-pixel loss
        # weight after densification ends. Requires --roi_keep_label_bin (ModelParams)
        # so the label stencils exist. -1 start = resolved to densify_until_iter.
        self.roi_late_refine = False
        self.roi_refine_label_mult = 2.0
        self.roi_refine_start_iter = -1
        self.roi_refine_ramp_iters = 300
        self.score_num_cameras = 10
        self.final_prune_interval = 3000
        self.final_prune_until_iter = 30000

        self.random_background = False
        self.optimizer_type = "default"
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
