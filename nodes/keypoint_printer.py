# ComfyUI_DWposeDeluxe/nodes/keypoint_printer.py

import torch
import numpy as np
import copy
import json

from comfy.model_management import InterruptProcessingException
from ..scripts import logger
from ..dwpose import util as dwpose_util
from .custom_options import DWOPOSE_CUSTOM_OPTIONS_TYPE
from ..node_configs import DWposeNodeBase


def detect_coordinate_format(data):
    for item in data:
        for person in item.get("people", []):
            pts = person.get("pose_keypoints_2d", [])
            if pts:
                for i in range(0, min(len(pts), 15), 3):
                    if abs(pts[i]) > 1.0 or abs(pts[i + 1]) > 1.0:
                        return "absolute"
    return "normalized"


class KeypointPrinter(DWposeNodeBase):
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("pose_image", "input_keypoint_info")
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


    def execute(self, keypoints, poses_to_print, show_body, show_feet, show_face, show_hands, custom_options=None):
        if not keypoints or not isinstance(keypoints, list):
            logger.warning("[PosePrinter] Keypoints data is empty or invalid. Returning empty image.")
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "Keypoints data is empty or invalid.")

        info_data = copy.deepcopy(keypoints)
        pose_info = ""
        if not info_data:
            pose_info = "Error: No keypoints data provided."
        else:
            try:
                pose_format = detect_coordinate_format(info_data)
                json_structure = "in-memory"

                confidence_variable = False
                for frame in info_data:
                    for person in frame.get("people", []):
                        for key in ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
                            pts = person.get(key)
                            if pts:
                                for i in range(2, len(pts), 3):
                                    if pts[i] not in [0.0, 1.0, 2.0]:
                                        confidence_variable = True
                                        break
                            if confidence_variable: break
                    if confidence_variable: break
                
                expected_counts = {"body": 18, "feet": 6, "face": 68, "left_hand": 21, "right_hand": 21}
                present_counts = {k: 0 for k in expected_counts}
                total_expected = {k: 0 for k in expected_counts}
                subset_exists = {k: False for k in expected_counts}

                for frame in info_data:
                    for person in frame.get("people", []):
                        pose_kpts = person.get("pose_keypoints_2d")
                        if pose_kpts:
                            subset_exists["body"] = True
                            total_expected["body"] += expected_counts["body"]
                            limit = min(len(pose_kpts), expected_counts["body"] * 3)
                            for i in range(2, limit, 3):
                                if pose_kpts[i-2] >= 0.0 and pose_kpts[i-1] >= 0.0 and pose_kpts[i] > 0.0:
                                    present_counts["body"] += 1

                            if len(pose_kpts) >= (expected_counts["body"] + expected_counts["feet"]) * 3:
                                subset_exists["feet"] = True
                                total_expected["feet"] += expected_counts["feet"]
                                start = expected_counts["body"] * 3
                                limit = min(len(pose_kpts), (expected_counts["body"] + expected_counts["feet"]) * 3)
                                for i in range(start + 2, limit, 3):
                                    if pose_kpts[i-2] >= 0.0 and pose_kpts[i-1] >= 0.0 and pose_kpts[i] > 0.0:
                                        present_counts["feet"] += 1
                        
                        subset_map = {"face": "face_keypoints_2d", "left_hand": "hand_left_keypoints_2d", "right_hand": "hand_right_keypoints_2d"}
                        for name, key in subset_map.items():
                            kpts = person.get(key)
                            if kpts:
                                subset_exists[name] = True
                                total_expected[name] += expected_counts[name]
                                limit = min(len(kpts), expected_counts[name] * 3)
                                for i in range(2, limit, 3):
                                    if kpts[i-2] >= 0.0 and kpts[i-1] >= 0.0 and kpts[i] > 0.0:
                                        present_counts[name] += 1
                
                info = {}
                for name in expected_counts:
                    if subset_exists[name] and total_expected[name] > 0:
                        percentage = (present_counts[name] / total_expected[name]) * 100
                        info[name] = f"yes ({percentage:.0f}%)"
                    else:
                        info[name] = "no"

                frame_count = len(info_data)
                poses_per_frame = [len(frame.get("people", [])) for frame in info_data if isinstance(frame, dict)]
                number_of_people = max(poses_per_frame) if poses_per_frame else 0

                first_frame_data = info_data[0] if info_data else {}
                width = first_frame_data.get("canvas_width", "n/a")
                height = first_frame_data.get("canvas_height", "n/a")

                pose_info = (
                    f"Canvas Width: {width}\n"
                    f"Canvas Height: {height}\n"
                    f"Format: {pose_format}\n"
                    f"Structure: {json_structure}\n"
                    f"Confidence: {'yes' if confidence_variable else 'no'}\n"
                    f"Body: {info['body']}\n"
                    f"Feet: {info['feet']}\n"
                    f"Face: {info['face']}\n"
                    f"Left Hand: {info['left_hand']}\n"
                    f"Right Hand: {info['right_hand']}\n"
                    f"Frame Count: {frame_count}\n"
                    f"Number of Poses: {number_of_people}"
                )
            except Exception as e:
                pose_info = f"Error generating pose info: {e}"

        keypoints = copy.deepcopy(keypoints)

        input_format = detect_coordinate_format(keypoints)

        if input_format == "absolute":
            logger.info("[PosePrinter] Detected absolute coordinates, converting to relative.")
            for frame in keypoints:
                width = frame.get('canvas_width')
                height = frame.get('canvas_height')
                if width is None or height is None or width == 0 or height == 0:
                    logger.error("[PosePrinter] Cannot normalize absolute keypoints without valid 'canvas_width' and 'canvas_height'.")
                    raise ValueError("Invalid keypoints data for normalization.")
                
                for person in frame.get("people", []):
                    for key in ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
                        if key in person:
                            pts = person[key]
                            for i in range(0, len(pts), 3):
                                pts[i] /= width
                                pts[i+1] /= height
        
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
            actual_options.update(copy.deepcopy(custom_options))
            logger.info(f"[PosePrinter] Received options from CustomOptions node.")
        else:
            logger.warning(f"[PosePrinter] CustomOptions node not connected, using default modifiers and thresholds.")

        for frame_keypoints_data in keypoints:
            width = frame_keypoints_data.get('canvas_width')
            height = frame_keypoints_data.get('canvas_height')

            if width is None or height is None:
                logger.error("[PosePrinter] Keypoints data is missing 'canvas_width' or 'canvas_height'. Cannot create canvas.")
                raise ValueError("Invalid keypoints data: missing canvas dimensions.")

            canvas_np = np.zeros((int(height), int(width), 3), dtype=np.uint8)

            people_to_process = frame_keypoints_data.get('people', [])
            if poses_to_print != -1:
                people_to_process = people_to_process[:poses_to_print]

            current_candidate = []
            current_subset = []
            all_hand_peaks = []
            all_lmks = []
            candidate_idx_offset = 0

            for person in people_to_process:
                if show_body:
                    pose_kpts_2d = np.array(person.get('pose_keypoints_2d', [])).reshape(-1, 3)
                    person_subset_row = [-1] * 25
                    
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

            if show_body and current_candidate and current_subset:
                canvas_np = dwpose_util.draw_bodypose(canvas_np, np.array(current_candidate), np.array(current_subset), show_feet, actual_options)
            
            if show_hands and all_hand_peaks:
                canvas_np = dwpose_util.draw_handpose(canvas_np, all_hand_peaks, actual_options)
            
            if show_face and all_lmks:
                canvas_np = dwpose_util.draw_facepose(canvas_np, all_lmks, actual_options)
            
            results_list.append(canvas_np)

        if not results_list:
            logger.warning("[PosePrinter] No poses were rendered. Returning empty image.")
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)

        output_tensor = torch.from_numpy(np.array(results_list).astype(np.float32) / 255.0)
        return (output_tensor, pose_info)

NODE_CLASS_MAPPINGS = {"KeypointPrinter": KeypointPrinter}
NODE_DISPLAY_NAME_MAPPINGS = {"KeypointPrinter": "DWpose Keypoint Printer"}