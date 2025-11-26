# ComfyUI_DWposeDeluxe/dwpose/__init__.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import json
from . import util
from .wholebody import Wholebody
from ..scripts import logger

# TensorRT Imports
tensorrt_available = False
try:
    import tensorrt as trt
    import torch
    tensorrt_available = True
except ImportError:
    pass

if tensorrt_available:
    from ..trt.trt_utilities import Engine

def draw_pose(pose, H, W, show_body=True, show_face=True, show_hands=True, show_feet=True, options={}):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    if show_body:
        if isinstance(candidate, np.ndarray) and isinstance(subset, np.ndarray) and candidate.size > 0 and subset.size > 0:
            canvas = util.draw_bodypose(canvas, candidate, subset, show_feet=show_feet, options=options)
        else:
            logger.warning(f"Skipping body/feet drawing due to empty candidate or subset")

    if show_hands and isinstance(hands, list) and len(hands) > 0:
        canvas = util.draw_handpose(canvas, hands, options=options)
    if show_face and isinstance(faces, list) and len(faces) > 0:
        canvas = util.draw_facepose(canvas, faces, options=options)
    if np.all(canvas == 0):
        pass
    else:
        pass
    return canvas

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)

def demo_postprocess(outputs, img_size, p6=False):
    grids = []
    expanded_strides = []
    strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs

def preprocess_det(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

from typing import List, Tuple

def preprocess_pose(
    img: np.ndarray, out_bbox, input_size: Tuple[int, int] = (192, 256)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Do preprocessing for RTMPose model inference."""
    img_shape = img.shape[:2]
    out_img, out_center, out_scale = [], [], []
    if len(out_bbox) == 0:
        out_bbox = [[0, 0, img_shape[1], img_shape[0]]]
    for i in range(len(out_bbox)):
        x0 = out_bbox[i][0]
        y0 = out_bbox[i][1]
        x1 = out_bbox[i][2]
        y1 = out_bbox[i][3]
        bbox = np.array([x0, y0, x1, y1])

        center, scale = bbox_xyxy2cs(bbox, padding=1.25)

        resized_img, scale = top_down_affine(input_size, scale, center, img)

        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        resized_img = (resized_img - mean) / std

        out_img.append(resized_img)
        out_center.append(center)
        out_scale.append(scale)

    return out_img, out_center, out_scale

def postprocess(
    outputs: List[np.ndarray],
    model_input_size: Tuple[int, int],
    center: Tuple[int, int],
    scale: Tuple[int, int],
    simcc_split_ratio: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Postprocess for RTMPose model output."""
    all_key = []
    all_score = []
    for i in range(len(outputs)):
        simcc_x, simcc_y = outputs[i]
        keypoints, scores = decode(simcc_x, simcc_y, simcc_split_ratio)

        keypoints = keypoints / model_input_size * scale[i] + center[i] - scale[i] / 2
        all_key.append(keypoints[0])
        all_score.append(scores[0])

    return np.array(all_key), np.array(all_score)

def bbox_xyxy2cs(bbox: np.ndarray, padding: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
    """Transform the bbox format from (x,y,w,h) into (center, scale)"""
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center, scale

def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
    """Extend the scale to match the given aspect ratio."""
    w, h = np.hsplit(bbox_scale, [1])
    bbox_scale = np.where(w > h * aspect_ratio, np.hstack([w, w / aspect_ratio]), np.hstack([h * aspect_ratio, h]))
    return bbox_scale

def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a point by an angle."""
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt

def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """To calculate the affine matrix, three pairs of points are required."""
    direction = a - b
    c = b + np.r_[-direction[1], direction[0]]
    return c

def get_warp_matrix(center: np.ndarray, scale: np.ndarray, rot: float, output_size: Tuple[int, int], shift: Tuple[float, float] = (0., 0.), inv: bool = False) -> np.ndarray:
    """Calculate the affine transformation matrix."""
    shift = np.array(shift)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0., src_w * -0.5]), rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return warp_mat

def top_down_affine(input_size: dict, bbox_scale: dict, bbox_center: dict, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get the bbox image as the model input by affine transform."""
    w, h = input_size
    warp_size = (int(w), int(h))

    bbox_scale = _fix_aspect_ratio(bbox_scale, aspect_ratio=w / h)

    center = bbox_center
    scale = bbox_scale
    rot = 0
    warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

    img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)

    return img, bbox_scale

def get_simcc_maximum(simcc_x: np.ndarray, simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from simcc representations."""
    N, K, Wx = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)

    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.] = -1

    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)

    return locs, vals

def decode(simcc_x: np.ndarray, simcc_y: np.ndarray, simcc_split_ratio) -> Tuple[np.ndarray, np.ndarray]:
    """Modulate simcc distribution with Gaussian."""
    keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)
    keypoints /= simcc_split_ratio

    return keypoints, scores

class DWposeDetector:
    def __init__(self, provider_type: str, det_model_path: str, pose_model_path: str, model_type: str = "ONNX",
                yolox_engine: 'Engine' = None, dwpose_engine: 'Engine' = None):
        self.model_type = model_type
        self.provider_type = provider_type
        self.det_engine = yolox_engine
        self.pose_engine = dwpose_engine
        self._det_context = None # For ONNX sessions
        self._pose_context = None # For ONNX sessions

        if self.model_type == "TensorRT":
            if not tensorrt_available:
                raise ImportError("TensorRT backend requested but 'tensorrt' module is not available\nPlease install TensorRT")
            logger.info(f"Initializing TensorRT backend for {provider_type}")

            if self.det_engine is None or self.pose_engine is None:
                raise ValueError("TensorRT engines must be provided for TensorRT model_type")
        else: # ONNX backend
            logger.info(f"Initializing ONNX backend for {provider_type}")

            self.pose_estimation = Wholebody(
                provider_type=provider_type,
                det_model_path=det_model_path,
                pose_model_path=pose_model_path
            )

    def activate_engines(self):
        if self.model_type == "TensorRT":
            if self.det_engine and self.pose_engine:
                self.det_engine.activate()
                self.det_engine.allocate_buffers()
                self.pose_engine.activate()
                self.pose_engine.allocate_buffers()

    def reset_engines(self):
        if self.model_type == "TensorRT":
            if self.det_engine and self.pose_engine:
                self.det_engine.reset()
                self.pose_engine.reset()

    def __call__(self, oriImg, show_body=True, show_face=True, show_hands=True, show_feet=True, poses_to_detect: int = 1, **render_options):
        H, W, C = oriImg.shape
        if self.model_type == "TensorRT":
            from .trt_inference import inference_detector_trt, inference_pose_trt
            
            cudaStream = torch.cuda.current_stream().cuda_stream

            det_result = inference_detector_trt(
                engine=self.det_engine, cudaStream=cudaStream, image_np_hwc=oriImg)
            keypoints, scores = inference_pose_trt(
                engine=self.pose_engine, cudaStream=cudaStream, out_bbox=det_result, image_np_hwc=oriImg)

            # Apply keypoint remapping logic for TRT path to match ONNX output
            keypoints_info = np.concatenate(
                (keypoints, scores[..., None]), axis=-1)
            neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
            neck_valid = np.logical_and(keypoints_info[:, 5, 2] > 0.3, keypoints_info[:, 6, 2] > 0.3)
            neck[:, 2] = neck_valid.astype(float)
            new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
            mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
            openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
            max_target_idx = max(openpose_idx)
            max_source_idx = max(mmpose_idx)
            if new_keypoints_info.shape[1] > max(max_target_idx, max_source_idx):
                 new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
            else:
                 logger.warning(f"Keypoint remapping indices out of bounds. Skipping remapping")
            keypoints_info = new_keypoints_info
            keypoints, scores = keypoints_info[..., :2], keypoints_info[..., 2]

        else: # ONNX backend
            oriImg = oriImg.copy()
            keypoints, scores = self.pose_estimation(oriImg)

        num_people = keypoints.shape[0] if isinstance(keypoints, np.ndarray) else 0
        if num_people == 0: return np.zeros((H, W, 3), dtype=np.uint8), {"people": []}

        # Filter poses based on poses_to_detect
        if poses_to_detect != -1 and num_people > poses_to_detect:
            person_areas = []
            for i in range(num_people):
                person_kpts = keypoints[i, :, :2]
                
                # Filter out keypoints with low confidence (score < 0.25)
                person_scores = scores[i, :]
                confident_kpts = person_kpts[person_scores > 0.25]

                if confident_kpts.shape[0] > 0:

                    # Calculate bounding box from confident keypoints
                    x_min, y_min = np.min(confident_kpts, axis=0)
                    x_max, y_max = np.max(confident_kpts, axis=0)
                    area = (x_max - x_min) * (y_max - y_min)
                    person_areas.append((area, i))
                else:
                    person_areas.append((0, i))

            # Sort by area in descending order
            person_areas.sort(key=lambda x: x[0], reverse=True)

            # Select top 'poses_to_detect' indices
            top_indices = [idx for area, idx in person_areas[:poses_to_detect]]
            
            # Filter keypoints and scores
            keypoints = keypoints[top_indices]
            scores = scores[top_indices]
            num_people = keypoints.shape[0]

        all_keypoints_pixel = keypoints
        all_scores = scores
        if W == 0 or H == 0: return np.zeros((H, W, 3), dtype=np.uint8)
        norm_factor = np.array([W, H], dtype=float).reshape(1, 1, 2)
        all_keypoints_norm = all_keypoints_pixel / norm_factor
        num_points_total = all_keypoints_norm.shape[1]

        candidate_list = []
        subset_list = []
        body_indices = list(range(18))
        feet_indices = list(range(18, 24))
        face_indices = list(range(24, 92))

        left_hand_indices = list(range(92, 113))
        right_hand_indices = list(range(113, 133))

        if show_feet: body_subset_indices = body_indices + feet_indices
        else: body_subset_indices = body_indices
        num_body_points = len(body_subset_indices)

        start_idx = 0
        confidence_threshold = 0.3
        points_above_thresh_count = 0

        for person_idx in range(num_people):
             person_kpts_norm = all_keypoints_norm[person_idx]
             person_scores = all_scores[person_idx]

             if len(body_subset_indices) > 0:
                  max_req_idx = max(body_subset_indices)
                  if max_req_idx < person_kpts_norm.shape[0]: candidate_list.append(person_kpts_norm[body_subset_indices])
                  else: candidate_list.append(np.zeros((num_body_points, person_kpts_norm.shape[1]), dtype=person_kpts_norm.dtype))
             subset_row = np.full(num_body_points, -1, dtype=int)
             person_points_above_thresh = 0
             for i, point_idx in enumerate(body_subset_indices):
                 if point_idx < len(person_scores) and person_scores[point_idx] > confidence_threshold:
                     subset_row[i] = start_idx + i; person_points_above_thresh += 1
             subset_list.append(subset_row)
             start_idx += num_body_points; points_above_thresh_count += person_points_above_thresh

        if not candidate_list: candidate_np = np.empty((0, 2), dtype=float); subset_np = np.empty((0, num_body_points), dtype=int)
        else: candidate_np = np.concatenate(candidate_list, axis=0).astype(float); subset_np = np.array(subset_list)

        faces_list = []
        face_points_above_thresh = 0
        if len(face_indices) > 0:
             max_face_idx = max(face_indices)
             for i in range(num_people):
                  if max_face_idx < all_keypoints_norm.shape[1]:
                       person_face_scores = all_scores[i, face_indices]
                       if np.any(person_face_scores > 0.1):
                            faces_list.append(all_keypoints_norm[i, face_indices]); face_points_above_thresh += np.sum(person_face_scores > 0.1)

        hands_list = []
        hand_points_above_thresh = 0

        # Check left hand
        if len(left_hand_indices) > 0:
            max_left_hand_idx = max(left_hand_indices)
            for i in range(num_people):
                 if max_left_hand_idx < all_keypoints_norm.shape[1]:
                      person_left_hand_scores = all_scores[i, left_hand_indices]

                      if np.any(person_left_hand_scores > 0.1):
                           hands_list.append(all_keypoints_norm[i, left_hand_indices])
                           hand_points_above_thresh += np.sum(person_left_hand_scores > 0.1)

        # Check right hand
        if len(right_hand_indices) > 0:
             max_right_hand_idx = max(right_hand_indices)
             for i in range(num_people):
                  if max_right_hand_idx < all_keypoints_norm.shape[1]:
                       person_right_hand_scores = all_scores[i, right_hand_indices]

                       if np.any(person_right_hand_scores > 0.1):
                            hands_list.append(all_keypoints_norm[i, right_hand_indices])
                            hand_points_above_thresh += np.sum(person_right_hand_scores > 0.1)

        if points_above_thresh_count == 0 and face_points_above_thresh == 0 and hand_points_above_thresh == 0:
             logger.warning(f"No keypoints found above confidence thresholds. Returning black canvas")
             return np.zeros((H, W, 3), dtype=np.uint8), {"people": []}


        bodies = dict(candidate=candidate_np, subset=subset_np)
        pose = dict(bodies=bodies, hands=hands_list, faces=faces_list)

        # Generate Keypoints JSON
        people_data = []
        for person_idx in range(num_people):
            person_keypoints_data = {}

            # Pose Keypoints
            if show_body:
                pose_kpts_2d = []
                indices_to_include = body_indices + (feet_indices if show_feet else [])
                for idx in indices_to_include:
                    x, y = all_keypoints_pixel[person_idx, idx]
                    score = all_scores[person_idx, idx]
                    pose_kpts_2d.extend([float(x), float(y), float(score)])
                person_keypoints_data["pose_keypoints_2d"] = pose_kpts_2d

            # Face Keypoints
            if show_face:
                face_kpts_2d = []
                for idx in face_indices:
                    x, y = all_keypoints_pixel[person_idx, idx]
                    score = all_scores[person_idx, idx]
                    face_kpts_2d.extend([float(x), float(y), float(score)])
                person_keypoints_data["face_keypoints_2d"] = face_kpts_2d

            # Left Hand Keypoints
            if show_hands: # show_hands already passed to draw_pose
                hand_left_kpts_2d = []
                for idx in left_hand_indices:
                    x, y = all_keypoints_pixel[person_idx, idx]
                    score = all_scores[person_idx, idx]
                    hand_left_kpts_2d.extend([float(x), float(y), float(score)])
                person_keypoints_data["hand_left_keypoints_2d"] = hand_left_kpts_2d

            # Right Hand Keypoints
            if show_hands: # show_hands already passed to draw_pose
                hand_right_kpts_2d = []
                for idx in right_hand_indices:
                    x, y = all_keypoints_pixel[person_idx, idx]
                    score = all_scores[person_idx, idx]
                    hand_right_kpts_2d.extend([float(x), float(y), float(score)])
                person_keypoints_data["hand_right_keypoints_2d"] = hand_right_kpts_2d

            people_data.append(person_keypoints_data)

        keypoints_json_data = {"people": people_data}

        final_canvas = draw_pose(pose, H, W, show_body=show_body, show_face=show_face, show_hands=show_hands, show_feet=show_feet, options=render_options)

        return final_canvas, keypoints_json_data