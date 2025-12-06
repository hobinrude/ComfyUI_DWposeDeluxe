# ComfyUI_DWposeDeluxe/nodes/keypoint_printer.py

import torch
import numpy as np

from comfy.model_management import InterruptProcessingException
from ..scripts import logger
from ..dwpose import util as dwpose_util
from .custom_options import DWOPOSE_CUSTOM_OPTIONS_TYPE

class KeypointPrinter:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pose_image",)
    FUNCTION = "execute"
    CATEGORY = "DWposeDeluxe"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keypoints": ("POSE_KEYPOINT",),
            },
            "optional": {
                "custom_options": (DWOPOSE_CUSTOM_OPTIONS_TYPE,),
                "poses_to_print": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "show_body": ("BOOLEAN", {"default": True}),
                "show_feet": ("BOOLEAN", {"default": True}),
                "show_face": ("BOOLEAN", {"default": True}),
                "show_hands": ("BOOLEAN", {"default": True}),
            }
        }

    def execute(self, keypoints, custom_options, poses_to_print, show_body, show_feet, show_face, show_hands):
        if not keypoints or not isinstance(keypoints, list):
            logger.warning("Keypoints data is empty or invalid. Returning empty image.")
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)

        # Coordinate Format Check
        for frame in keypoints:
            for person in frame.get("people", []):
                for key in ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
                    kpts = person.get(key, [])
                    # Check only a few points for efficiency
                    for i in range(0, min(len(kpts), 9), 3):
                        if kpts[i] > 1.0 or kpts[i+1] > 1.0:
                            logger.error("Keypoints are in absolute (pixel) format. Please use a Keypoint Converter node to normalize them before using the printer.")
                            raise InterruptProcessingException("Keypoints must be in normalized format.")
        
        results_list = []
        actual_options = {}
        default_options = {
            "body_dot_size_modifier": 0, "body_line_thickness_modifier": 0,
            "hand_dot_size_modifier": 0, "hand_line_thickness_modifier": 0,
            "face_dot_size_modifier": 0,
            "pose_threshold": 0.25,
            "body_threshold": 0.30,
            "face_threshold": 0.10,
            "hand_threshold": 0.10,
            "face_padding": 0.0,
        }
        actual_options.update(default_options)

        if custom_options is not None:
            actual_options.update(custom_options)
            logger.info(f"KeypointPrinter: Received options from CustomOptions node.")
        else:
            logger.warning(f"KeypointPrinter: CustomOptions node not connected, using default modifiers and thresholds.")

        for frame_keypoints_data in keypoints:
            width = frame_keypoints_data.get('canvas_width')
            height = frame_keypoints_data.get('canvas_height')

            if width is None or height is None:
                logger.error("Keypoints data is missing 'canvas_width' or 'canvas_height'. Cannot create canvas.")
                raise ValueError("Invalid keypoints data: missing canvas dimensions.")

            canvas_np = np.zeros((height, width, 3), dtype=np.uint8)

            people_to_process = frame_keypoints_data.get('people', [])
            if poses_to_print != -1:
                people_to_process = people_to_process[:poses_to_print]

            # Data Transformation
            current_candidate = []
            current_subset = []
            all_hand_peaks = []
            all_lmks = []
            candidate_idx_offset = 0

            for person in people_to_process:
                if show_body:
                    pose_kpts_2d = np.array(person.get('pose_keypoints_2d', [])).reshape(-1, 3)
                    person_subset_row = [-1] * 25 # DWpose uses up to 25 keypoints for body/feet
                    
                    for kpt_idx, (x, y, conf) in enumerate(pose_kpts_2d):
                        if kpt_idx < 25 and conf > 0:
                            current_candidate.append([x, y, conf])
                            person_subset_row[kpt_idx] = candidate_idx_offset
                            candidate_idx_offset += 1
                    current_subset.append(person_subset_row)

                if show_hands:
                    left_hand_kpts = np.array(person.get('hand_left_keypoints_2d', [])).reshape(-1, 3)
                    normalized_left_hand_kpts = [[x, y] for x, y, conf in left_hand_kpts if conf > 0]
                    if normalized_left_hand_kpts:
                        all_hand_peaks.append(normalized_left_hand_kpts)

                    right_hand_kpts = np.array(person.get('hand_right_keypoints_2d', [])).reshape(-1, 3)
                    normalized_right_hand_kpts = [[x, y] for x, y, conf in right_hand_kpts if conf > 0]
                    if normalized_right_hand_kpts:
                        all_hand_peaks.append(normalized_right_hand_kpts)
                
                if show_face:
                    face_kpts = np.array(person.get('face_keypoints_2d', [])).reshape(-1, 3)
                    normalized_face_kpts = [[x, y] for x, y, conf in face_kpts if conf > 0]
                    if normalized_face_kpts:
                        all_lmks.append(normalized_face_kpts)

            # Rendering
            if show_body and current_candidate and current_subset:
                canvas_np = dwpose_util.draw_bodypose(canvas_np, np.array(current_candidate), np.array(current_subset), show_feet, actual_options)
            
            if show_hands and all_hand_peaks:
                canvas_np = dwpose_util.draw_handpose(canvas_np, all_hand_peaks, actual_options)
            
            if show_face and all_lmks:
                canvas_np = dwpose_util.draw_facepose(canvas_np, all_lmks, actual_options)
            
            results_list.append(canvas_np)

        if not results_list:
            logger.warning("No poses were rendered. Returning empty image.")
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)

        output_tensor = torch.from_numpy(np.array(results_list).astype(np.float32) / 255.0)
        return (output_tensor,)

NODE_CLASS_MAPPINGS = {"KeypointPrinter": KeypointPrinter}

NODE_DISPLAY_NAME_MAPPINGS = {"KeypointPrinter": "DWpose Keypoint Printer"}