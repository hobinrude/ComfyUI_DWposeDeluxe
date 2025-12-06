# ComfyUI_DWposeDeluxe/dwpose/__init__.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import json
import onnxruntime as ort

from . import util
from .detection import inference_detector
from .estimation import inference_pose
from ..scripts import logger

tensorrt_available = False
try:
    import tensorrt as trt
    import torch
    tensorrt_available = True
except ImportError:
    pass

if tensorrt_available:
    from ..trt.trt_utilities import Engine

class DWposeDetector:
    def __init__(self, provider_type: str, det_model_path: str, pose_model_path: str, model_type: str = "ONNX",
                yolox_engine: 'Engine' = None, dwpose_engine: 'Engine' = None):
        self.model_type = model_type
        self.provider_type = provider_type
        self.det_session = None
        self.pose_session = None

        if self.model_type == "TensorRT":
            if not tensorrt_available:
                raise ImportError("TensorRT backend requested but 'tensorrt' module is not available. Please install TensorRT.")
            logger.info(f"Initializing TensorRT backend for {provider_type}")
            if yolox_engine is None or dwpose_engine is None:
                raise ValueError("TensorRT engines must be provided for TensorRT model_type.")
            self.det_session = yolox_engine
            self.pose_session = dwpose_engine
        else: # ONNX
            logger.info(f"Initializing ONNX backend for {provider_type}")
            providers = ['CPUExecutionProvider'] if provider_type == 'CPU' else ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            if not os.path.exists(det_model_path):
                raise FileNotFoundError(f"Detector model not found at: {det_model_path}")
            if not os.path.exists(pose_model_path):
                raise FileNotFoundError(f"Pose model not found at: {pose_model_path}")

            try:
                self.det_session = ort.InferenceSession(path_or_bytes=det_model_path, providers=providers)
                self.pose_session = ort.InferenceSession(path_or_bytes=pose_model_path, providers=providers)
                logger.info(f"ONNX sessions created successfully")
            except Exception as e:
                logger.error(f"Failed to create ONNX session: {e}")
                raise e

    def activate_engines(self):
        if self.model_type == "TensorRT":
            if self.det_session and self.pose_session:
                self.det_session.activate()
                self.det_session.allocate_buffers()
                self.pose_session.activate()
                self.pose_session.allocate_buffers()

    def reset_engines(self):
        if self.model_type == "TensorRT":
            if self.det_session and self.pose_session:
                self.det_session.reset()
                self.pose_session.reset()

    def __call__(self, oriImg, show_body=True, show_face=True, show_hands=True, show_feet=True, poses_to_detect: int = 1, pose_threshold: float = 0.25, body_threshold: float = 0.3, face_threshold: float = 0.1, hand_threshold: float = 0.1, **render_options):
        H, W, C = oriImg.shape
        
        neck_validity = render_options.get('neck_validity', 0.3)
        nms_threshold = render_options.get('nms_threshold', 0.45)
        score_threshold = render_options.get('score_threshold', 0.10)

        # Unified Detection
        det_result = inference_detector(self.det_session, oriImg, self.model_type, nms_threshold, score_threshold)

        if det_result is None or not np.any(det_result):
            return np.zeros((H, W, 3), dtype=np.uint8), {"people": []}

        # Unified Estimation
        keypoints, scores = inference_pose(self.pose_session, det_result, oriImg, self.model_type)

        if keypoints is None or scores is None or keypoints.shape[0] == 0:
            return np.zeros((H, W, 3), dtype=np.uint8), {"people": []}

        # Unified Post-processing (Neck and Remapping)
        keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        neck_valid = np.logical_and(keypoints_info[:, 5, 2] > neck_validity, keypoints_info[:, 6, 2] > neck_validity)
        neck[:, 2] = neck_valid.astype(float)
        
        new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
        
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        
        max_target_idx = max(openpose_idx)
        if new_keypoints_info.shape[1] > max_target_idx:
             new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
        else:
             logger.warning(f"Keypoint remapping indices out of bounds. Skipping remapping")
        
        keypoints_info = new_keypoints_info
        keypoints, scores = keypoints_info[..., :2], keypoints_info[..., 2]

        # Common filtering and data formatting logic
        num_people = keypoints.shape[0]
        if poses_to_detect != -1 and num_people > poses_to_detect:
            person_areas = []
            for i in range(num_people):
                person_kpts = keypoints[i, :, :2]
                person_scores = scores[i, :]
                confident_kpts = person_kpts[person_scores > pose_threshold]

                if confident_kpts.shape[0] > 0:
                    x_min, y_min = np.min(confident_kpts, axis=0)
                    x_max, y_max = np.max(confident_kpts, axis=0)
                    area = (x_max - x_min) * (y_max - y_min)
                    person_areas.append((area, i))
                else:
                    person_areas.append((0, i))

            person_areas.sort(key=lambda x: x[0], reverse=True)
            top_indices = [idx for area, idx in person_areas[:poses_to_detect]]
            keypoints = keypoints[top_indices]
            scores = scores[top_indices]
            num_people = keypoints.shape[0]

        all_keypoints_pixel = keypoints
        all_scores = scores
        if W == 0 or H == 0: return np.zeros((H, W, 3), dtype=np.uint8), {"people": []}
        norm_factor = np.array([W, H], dtype=float).reshape(1, 1, 2)
        all_keypoints_norm = all_keypoints_pixel / norm_factor

        body_indices = list(range(18))
        feet_indices = list(range(18, 24))
        face_indices = list(range(24, 92))
        left_hand_indices = list(range(92, 113))
        right_hand_indices = list(range(113, 133))

        candidate_list = []
        subset_list = []
        if show_feet: body_subset_indices = body_indices + feet_indices
        else: body_subset_indices = body_indices
        num_body_points = len(body_subset_indices)
        start_idx = 0
        points_above_thresh_count = 0

        for person_idx in range(num_people):
             person_kpts_norm = all_keypoints_norm[person_idx]
             person_scores = all_scores[person_idx]
             if len(body_subset_indices) > 0:
                  max_req_idx = max(body_subset_indices)
                  if max_req_idx < person_kpts_norm.shape[0]:
                      candidate_list.append(person_kpts_norm[body_subset_indices])
                  else:
                      candidate_list.append(np.zeros((num_body_points, person_kpts_norm.shape[1]), dtype=person_kpts_norm.dtype))
             
             subset_row = np.full(num_body_points, -1, dtype=int)
             person_points_above_thresh = 0
             for i, point_idx in enumerate(body_subset_indices):
                 if point_idx < len(person_scores) and person_scores[point_idx] > body_threshold:
                     subset_row[i] = start_idx + i
                     person_points_above_thresh += 1
             subset_list.append(subset_row)
             start_idx += num_body_points
             points_above_thresh_count += person_points_above_thresh

        if not candidate_list: candidate_np = np.empty((0, 2), dtype=float)
        else: candidate_np = np.concatenate(candidate_list, axis=0).astype(float)
        subset_np = np.array(subset_list)

        faces_list = []
        face_points_above_thresh = 0
        if len(face_indices) > 0:
             max_face_idx = max(face_indices)
             for i in range(num_people):
                  if max_face_idx < all_keypoints_norm.shape[1]:
                       person_face_scores = all_scores[i, face_indices]
                       if np.any(person_face_scores > face_threshold):
                            faces_list.append(all_keypoints_norm[i, face_indices])
                            face_points_above_thresh += np.sum(person_face_scores > face_threshold)

        hands_list = []
        hand_points_above_thresh = 0
        if len(left_hand_indices) > 0:
            max_left_hand_idx = max(left_hand_indices)
            for i in range(num_people):
                 if max_left_hand_idx < all_keypoints_norm.shape[1]:
                      person_left_hand_scores = all_scores[i, left_hand_indices]
                      if np.any(person_left_hand_scores > hand_threshold):
                           hands_list.append(all_keypoints_norm[i, left_hand_indices])
                           hand_points_above_thresh += np.sum(person_left_hand_scores > hand_threshold)
        if len(right_hand_indices) > 0:
             max_right_hand_idx = max(right_hand_indices)
             for i in range(num_people):
                  if max_right_hand_idx < all_keypoints_norm.shape[1]:
                       person_right_hand_scores = all_scores[i, right_hand_indices]
                       if np.any(person_right_hand_scores > hand_threshold):
                            hands_list.append(all_keypoints_norm[i, right_hand_indices])
                            hand_points_above_thresh += np.sum(person_right_hand_scores > hand_threshold)

        if points_above_thresh_count == 0 and face_points_above_thresh == 0 and hand_points_above_thresh == 0:
             logger.warning(f"No keypoints found above confidence thresholds. Returning black canvas")
             return np.zeros((H, W, 3), dtype=np.uint8), {"people": []}

        bodies = dict(candidate=candidate_np, subset=subset_np)
        pose = dict(bodies=bodies, hands=hands_list, faces=faces_list)

        people_data = []
        for person_idx in range(num_people):
            person_keypoints_data = {}
            if show_body:
                pose_kpts_2d = []
                indices_to_include = body_indices + (feet_indices if show_feet else [])
                for idx in indices_to_include:
                    x, y = all_keypoints_pixel[person_idx, idx]
                    score = all_scores[person_idx, idx]
                    pose_kpts_2d.extend([float(x), float(y), float(score)])
                person_keypoints_data["pose_keypoints_2d"] = pose_kpts_2d

            face_kpts_2d = []
            face_kpts_for_bbox = []
            face_keypoint_map_indices = list(range(24, 92))
            for idx in face_keypoint_map_indices:
                if idx < all_keypoints_pixel.shape[1]:
                    x, y = all_keypoints_pixel[person_idx, idx]
                    score = all_scores[person_idx, idx]
                else:
                    x, y, score = 0, 0, 0
                if score > face_threshold:
                    face_kpts_for_bbox.append((x, y))
                face_kpts_2d.extend([float(x), float(y), float(score)])

            if show_face:
                person_keypoints_data["face_keypoints_2d"] = face_kpts_2d
            
            if face_kpts_for_bbox:
                x_coords, y_coords = zip(*face_kpts_for_bbox)
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                person_keypoints_data["face_box"] = [float(x_min), float(y_min), float(x_max), float(y_max)]

            if show_hands:
                hand_left_kpts_2d = []
                for idx in left_hand_indices:
                    x, y = all_keypoints_pixel[person_idx, idx]
                    score = all_scores[person_idx, idx]
                    hand_left_kpts_2d.extend([float(x), float(y), float(score)])
                person_keypoints_data["hand_left_keypoints_2d"] = hand_left_kpts_2d

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
    
    return canvas
