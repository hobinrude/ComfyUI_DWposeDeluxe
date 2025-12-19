# ComfyUI_DWposeDeluxe/nodes/keypoint_printer.py

import torch
import numpy as np
import copy
import json
import cv2

from comfy.model_management import InterruptProcessingException
from ..scripts import logger
from ..scripts.progress import progress
from ..scripts.keypoint_utils import UnifiedKeypointHandler
from ..dwpose import util as dwpose_util
from .custom_options import DWOPOSE_CUSTOM_OPTIONS_TYPE
from ..node_configs import DWposeNodeBase

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
                "image": ("IMAGE",),
                "poses_to_print": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "background": (["none", "scale", "crop/pad"], {"default": "none"}),
                "brightness": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "custom_options": (DWOPOSE_CUSTOM_OPTIONS_TYPE,),
                "show_body": ("BOOLEAN", {"default": True}),
                "show_feet": ("BOOLEAN", {"default": True}),
                "show_face": ("BOOLEAN", {"default": True}),
                "show_hands": ("BOOLEAN", {"default": True}),
            }
        }

    def execute(self, keypoints, poses_to_print, show_body, show_feet, show_face, show_hands, 
                image=None, background="none", brightness=0.50, custom_options=None):

        if not keypoints or not isinstance(keypoints, list):
            logger.warning("[PosePrinter] Keypoints data is empty or invalid. Returning empty image.")
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "Keypoints data is empty or invalid.")

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
            logger.info(f"[PosePrinter] Received options from CustomOptions node:")
            print(f" Body dot size modifier       : {actual_options['body_dot_size_modifier']}")
            print(f" Body bone thickness modifier : {actual_options['body_line_thickness_modifier']}")
            print(f" Hand dot size modifier       : {actual_options['hand_dot_size_modifier']}")
            print(f" Hand bone thickness modifier : {actual_options['hand_line_thickness_modifier']}")
            print(f" Face dot size modifier       : {actual_options['face_dot_size_modifier']}")
            print(f" Pose threshold               : {actual_options['pose_threshold']}")
            print(f" Body threshold               : {actual_options['body_threshold']}")
            print(f" Face threshold               : {actual_options['face_threshold']}")
            print(f" Hand threshold               : {actual_options['hand_threshold']}")
            print(f" Face padding                 : {actual_options['face_padding']}")
            print(f" Pose opacity                 : {actual_options.get('pose_opacity', 0.6)}")
        else:
            logger.warning(f"[PosePrinter] CustomOptions node not connected, using default modifiers and thresholds.")

        info_data = copy.deepcopy(keypoints)
        pose_info = ""
        if not info_data:
            pose_info = "Error: No keypoints data provided."
        else:
            try:
                pose_format = UnifiedKeypointHandler.detect_format(info_data)
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
                            limit = min(len(pose_kpts) // 3, 25) 
                            for kpt_idx in range(limit):
                                if pose_kpts[kpt_idx*3+2] > 0:
                                    if kpt_idx < 18: present_counts["body"] += 1
                                    else: present_counts["feet"] += 1
                            if limit > 18: subset_exists["feet"] = True
                        
                        subset_map = {"face": "face_keypoints_2d", "left_hand": "hand_left_keypoints_2d", "right_hand": "hand_right_keypoints_2d"}
                        for name, key in subset_map.items():
                            kpts = person.get(key)
                            if kpts:
                                subset_exists[name] = True
                                total_expected[name] += expected_counts[name]
                                limit = min(len(kpts) // 3, expected_counts[name])
                                for kpt_idx in range(limit):
                                    if kpts[kpt_idx*3+2] > 0:
                                        present_counts[name] += 1
                
                info = {}
                for name in expected_counts:
                    if subset_exists[name] and total_expected[name] > 0:
                        percentage = (present_counts[name] / (total_expected[name] * len(info_data))) * 100
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

        input_format = UnifiedKeypointHandler.detect_format(keypoints)

        # Always work with normalized coordinates internally for drawing
        if input_format == "absolute":
            first_frame = keypoints[0]
            width = first_frame.get('canvas_width')
            height = first_frame.get('canvas_height')
            if width and height:
                keypoints = UnifiedKeypointHandler.to_normalized(keypoints, width, height)
            else:
                logger.error("[PosePrinter] Cannot normalize absolute keypoints without valid 'canvas_width' and 'canvas_height'.")
                raise ValueError("Invalid keypoints data for normalization.")
        
        results_list = []
        
        first_keypoint_frame = keypoints[0]
        kp_canvas_w = int(first_keypoint_frame.get('canvas_width', 512))
        kp_canvas_h = int(first_keypoint_frame.get('canvas_height', 512))
        
        num_keypoint_frames = len(keypoints)
        num_image_frames = image.shape[0] if image is not None else 0
        
        total_frames_to_process = num_keypoint_frames
        if image is not None:
            if num_keypoint_frames != num_image_frames:
                logger.warning(f"[PosePrinter] Mismatch in frame counts: Keypoints ({num_keypoint_frames}) vs. Image ({num_image_frames}). Processing will stop at the shortest sequence.")
            total_frames_to_process = min(num_keypoint_frames, num_image_frames)

        pbar = progress(total_frames_to_process, label="Render")

        # Thresholds
        body_threshold = actual_options.get("body_threshold", 0.3)
        face_threshold = actual_options.get("face_threshold", 0.1)
        hand_threshold = actual_options.get("hand_threshold", 0.1)

        for frame_idx in range(total_frames_to_process):
            frame_keypoints_data = keypoints[frame_idx]
            
            current_canvas = np.zeros((kp_canvas_h, kp_canvas_w, 3), dtype=np.uint8)

            if image is not None and background != "none":
                bg_image_tensor = image[frame_idx]
                bg_image_np = (bg_image_tensor.cpu().numpy() * 255).astype(np.uint8)
                
                if bg_image_np.shape[2] == 4: bg_image_np = cv2.cvtColor(bg_image_np, cv2.COLOR_RGBA2RGB)
                elif bg_image_np.shape[2] == 1: bg_image_np = np.repeat(bg_image_np, 3, axis=2)
                elif bg_image_np.shape[2] == 3: pass
                else: logger.warning("[PosePrinter] Unsupported image channel format for background, using black canvas.")
                
                img_h, img_w, _ = bg_image_np.shape

                if background == "scale":
                    processed_bg = cv2.resize(bg_image_np, (kp_canvas_w, kp_canvas_h), interpolation=cv2.INTER_AREA)
                elif background == "crop/pad":

                    kp_aspect = kp_canvas_w / kp_canvas_h
                    img_aspect = img_w / img_h

                    if kp_aspect > img_aspect:

                        new_img_h = int(img_w / kp_aspect)
                        if new_img_h > img_h:
                            padding_h = (new_img_h - img_h) // 2
                            processed_bg = np.pad(bg_image_np, ((padding_h, padding_h), (0,0), (0,0)), mode='constant', constant_values=0)
                            processed_bg = cv2.resize(processed_bg, (kp_canvas_w, kp_canvas_h), interpolation=cv2.INTER_AREA)
                        else:
                            crop_h = (img_h - new_img_h) // 2
                            cropped_img = bg_image_np[crop_h:img_h-crop_h, :, :]
                            processed_bg = cv2.resize(cropped_img, (kp_canvas_w, kp_canvas_h), interpolation=cv2.INTER_AREA)

                    else:
                        new_img_w = int(img_h * kp_aspect)
                        if new_img_w > img_w:
                            padding_w = (new_img_w - img_w) // 2
                            processed_bg = np.pad(bg_image_np, ((0,0), (padding_w, padding_w), (0,0)), mode='constant', constant_values=0)
                            processed_bg = cv2.resize(processed_bg, (kp_canvas_w, kp_canvas_h), interpolation=cv2.INTER_AREA)
                        else:
                            crop_w = (img_w - new_img_w) // 2
                            cropped_img = bg_image_np[:, crop_w:img_w-crop_w, :]
                            processed_bg = cv2.resize(cropped_img, (kp_canvas_w, kp_canvas_h), interpolation=cv2.INTER_AREA)
                
                processed_bg_float = processed_bg.astype(np.float32) / 255.0
                processed_bg_float = processed_bg_float * brightness
                current_canvas = (processed_bg_float * 255).astype(np.uint8)


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
                    # Use granular threshold and ignore negative coordinates
                    filtered_pose_kpts = []
                    for kpt in pose_kpts_2d:
                        if kpt[2] >= body_threshold and kpt[0] >= 0 and kpt[1] >= 0:
                            filtered_pose_kpts.append(kpt)
                        else:
                            filtered_pose_kpts.append([0.0, 0.0, 0.0]) # Mark as invisible
                    
                    filtered_pose_kpts = filtered_pose_kpts[:25]
                    
                    person_subset_row = [-1] * 25 
                    
                    for kpt_idx, (x, y, conf) in enumerate(filtered_pose_kpts):
                        if conf > 0:
                            current_candidate.append([x, y, conf])
                            person_subset_row[kpt_idx] = candidate_idx_offset
                            candidate_idx_offset += 1
                    current_subset.append(person_subset_row)

                if show_hands:
                    left_hand_kpts = np.array(person.get('hand_left_keypoints_2d', [])).reshape(-1, 3)
                    filtered_left_hand_kpts = [[x, y] for x, y, conf in left_hand_kpts if conf >= hand_threshold and x >= 0 and y >= 0]
                    if filtered_left_hand_kpts:
                        all_hand_peaks.append(filtered_left_hand_kpts)

                    right_hand_kpts = np.array(person.get('hand_right_keypoints_2d', [])).reshape(-1, 3)
                    filtered_right_hand_kpts = [[x, y] for x, y, conf in right_hand_kpts if conf >= hand_threshold and x >= 0 and y >= 0]
                    if filtered_right_hand_kpts:
                        all_hand_peaks.append(filtered_right_hand_kpts)
                
                if show_face:
                    face_kpts = np.array(person.get('face_keypoints_2d', [])).reshape(-1, 3)
                    filtered_face_kpts = [[x, y] for x, y, conf in face_kpts if conf >= face_threshold and x >= 0 and y >= 0]
                    if filtered_face_kpts:
                        all_lmks.append(filtered_face_kpts)

            if show_body and current_candidate and current_subset:
                pose_opacity = actual_options.get("pose_opacity", 0.6)
                current_canvas = dwpose_util.draw_bodypose(current_canvas, np.array(current_candidate), np.array(current_subset), show_feet, actual_options, pose_opacity=pose_opacity)
            
            if show_hands and all_hand_peaks:
                current_canvas = dwpose_util.draw_handpose(current_canvas, all_hand_peaks, actual_options)
            
            if show_face and all_lmks:
                current_canvas = dwpose_util.draw_facepose(current_canvas, all_lmks, actual_options)
            
            results_list.append(current_canvas)
            pbar.step()
        pbar.finish()

        if not results_list:
            logger.warning("[PosePrinter] No poses were rendered. Returning empty image.")
            return (torch.zeros((1, kp_canvas_h, kp_canvas_w, 3), dtype=torch.float32), pose_info)

        output_tensor = torch.from_numpy(np.array(results_list).astype(np.float32) / 255.0)
        return (output_tensor, pose_info)

NODE_CLASS_MAPPINGS = {"KeypointPrinter": KeypointPrinter}
NODE_DISPLAY_NAME_MAPPINGS = {"KeypointPrinter": "DWpose Keypoint Printer"}
