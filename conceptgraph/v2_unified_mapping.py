"""One native CG mapping update; all native backend functions/thresholds retained."""
import numpy as np

from conceptgraph.slam.utils import gobs_to_detection_list, denoise_objects, filter_objects, merge_objects
from conceptgraph.slam.mapping import (compute_spatial_similarities, compute_visual_similarities,
                                       aggregate_similarities, merge_detections_to_objects)


def step(cfg, objects, index, color_path, image_rgb, depth_array, cam_K, pose, gobs):
    if cfg.use_contain_number or not cfg.skip_bg:
        raise ValueError('This control preserves the original class-agnostic skip_bg configuration')
    detections, background = gobs_to_detection_list(
        cfg=cfg, image=image_rgb, depth_array=depth_array, cam_K=cam_K, idx=index,
        gobs=gobs, trans_pose=pose, class_names=['item'], BG_CLASSES=['wall','floor','ceiling'],
        color_path=str(color_path))
    if background:
        raise ValueError('No synthesized background ID may enter the object map')
    if len(detections) == 0:
        return objects, 0
    if len(objects) == 0:
        objects.extend(detections)
        return objects, len(detections)
    spatial = compute_spatial_similarities(cfg, detections, objects)
    visual = compute_visual_similarities(cfg, detections, objects)
    similarities = aggregate_similarities(cfg, spatial, visual)
    similarities[similarities < cfg.sim_threshold] = float('-inf')
    objects = merge_detections_to_objects(cfg, detections, objects, similarities)
    # Early-return behavior and modulo global-index scheduling are native.
    if cfg.denoise_interval > 0 and (index+1) % cfg.denoise_interval == 0:
        objects = denoise_objects(cfg, objects)
    if cfg.filter_interval > 0 and (index+1) % cfg.filter_interval == 0:
        objects = filter_objects(cfg, objects)
    if cfg.merge_interval > 0 and (index+1) % cfg.merge_interval == 0:
        objects = merge_objects(cfg, objects)
    return objects, len(detections)
