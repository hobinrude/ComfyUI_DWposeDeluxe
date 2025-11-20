# ComfyUI_DWposeDeluxe/dwpose/wholebody.py

import cv2
import numpy as np
import onnxruntime as ort
import os
import inspect

# Keep these imports relative if onnxdet/onnxpose are in the same directory
from .onnxdet import inference_detector
from .onnxpose import inference_pose

class Wholebody:
    # Accepts provider and model paths
    def __init__(self, det_model_path: str, pose_model_path: str, provider_type: str):

        if not os.path.exists(det_model_path):
            raise FileNotFoundError(f"[WholeBody][{fg('red')}ERROR{attr('reset')}] Detector model not found at: {det_model_path}")
        if not os.path.exists(pose_model_path):
            raise FileNotFoundError(f"[WholeBody][{fg('red')}ERROR{attr('reset')}]Pose model not found at: {pose_model_path}")

        providers = ['CPUExecutionProvider'] if provider_type == 'CPU' else ['CUDAExecutionProvider', 'CPUExecutionProvider']

        try:
             self.session_det = ort.InferenceSession(path_or_bytes=det_model_path, providers=providers)
             self.session_pose = ort.InferenceSession(path_or_bytes=pose_model_path, providers=providers)
             print("[WholeBody][{fg('green')}INFO{attr('reset')}] ONNX sessions created successfully.")
        except Exception as e:
             print(f"[WholeBody][{fg('red')}ERROR{attr('reset')}] Failed to create ONNX session: {e}")
             raise e

    def __call__(self, oriImg):

        try:
            det_result = inference_detector(self.session_det, oriImg)
        except TypeError as e:
            raise e
        except Exception as e:
            raise e

        if det_result is None or not np.any(det_result):
             return np.empty((0, 133, 2), dtype=np.float32), np.empty((0, 133), dtype=np.float32)
        try:
            keypoints, scores = inference_pose(self.session_pose, det_result, oriImg)
        except Exception as e:
             raise e
        if keypoints is None or scores is None or keypoints.shape[0] == 0:
             return np.empty((0, 133, 2), dtype=np.float32), np.empty((0, 133), dtype=np.float32)
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
             print(f"[WholeBody][{fg('yellow')}WARNING{attr('reset')}] Keypoint remapping indices out of bounds. Skipping remapping.")
        keypoints_info = new_keypoints_info
        keypoints_final, scores_final = keypoints_info[..., :2], keypoints_info[..., 2]
        return keypoints_final, scores_final