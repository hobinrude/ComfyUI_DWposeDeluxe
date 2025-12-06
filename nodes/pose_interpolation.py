# ComfyUI_DWposeDeluxe/nodes/pose_interpolation.py

import torch
import numpy as np

from comfy.model_management import InterruptProcessingException
from ..scripts import logger

class PoseInterpolation:
    @staticmethod
    def _get_pose_from_dataset(dataset, frame_selector):
        if not dataset:
            return None, None, None

        selected_frame_data = None
        if frame_selector == "first":
            selected_frame_data = dataset[0]
        elif frame_selector == "last":
            selected_frame_data = dataset[-1]
        
        if not selected_frame_data or not selected_frame_data.get("people"):
            return None, None, None

        people = selected_frame_data["people"]
        canvas_width = selected_frame_data.get("canvas_width")
        canvas_height = selected_frame_data.get("canvas_height")

        if not canvas_width or not canvas_height:
            logger.error(f"Selected frame for '{frame_selector}' is missing canvas dimensions. Use KeypointConverter to fix it.")
            return None, None, None

        if len(people) > 1:
            logger.warning(f"Multiple poses found in the '{frame_selector}' frame. Selecting the biggest pose for interpolation.")
            
            biggest_pose_area = -1
            biggest_pose_data = None

            for person in people:
                pose_kpts_2d = np.array(person.get("pose_keypoints_2d", [])).reshape(-1, 3)
                if len(pose_kpts_2d) > 0:
                    x_coords = pose_kpts_2d[:, 0]
                    y_coords = pose_kpts_2d[:, 1]

                    # Filter out keypoints with 0 confidence before calculating bounds
                    valid_kpts_indices = np.where(pose_kpts_2d[:, 2] > 0)[0]
                    if len(valid_kpts_indices) == 0:
                        continue # No valid keypoints for this person

                    valid_x = x_coords[valid_kpts_indices]
                    valid_y = y_coords[valid_kpts_indices]

                    min_x, max_x = np.min(valid_x), np.max(valid_x)
                    min_y, max_y = np.min(valid_y), np.max(valid_y)

                    # Bounding box coordinates are normalized, convert to pixels for area calculation
                    # The canvas width/height are available here
                    bbox_width = (max_x - min_x) * canvas_width
                    bbox_height = (max_y - min_y) * canvas_height
                    
                    area = bbox_width * bbox_height
                    
                    if area > biggest_pose_area:
                        biggest_pose_area = area
                        biggest_pose_data = person
            
            if biggest_pose_data is None:
                logger.warning(f"Could not find a valid pose in the '{frame_selector}' frame for interpolation.")
                return None, None, None
            return biggest_pose_data, canvas_width, canvas_height
        
        return people[0], canvas_width, canvas_height # Only one person, return it directly

    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("keypoints",)
    FUNCTION = "execute"
    CATEGORY = "DWposeDeluxe"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keypoints_1": ("POSE_KEYPOINT",),
                "keypoints_2": ("POSE_KEYPOINT",),
                "keypoints_1_frame": (["first", "last"], {"default": "first"}),
                "keypoints_2_frame": (["first", "last"], {"default": "last"}),
                "frame_count": ("INT", {"default": 1, "min": 1, "max": 99, "step": 1}),
            }
        }

    def execute(self, keypoints_1, keypoints_2, keypoints_1_frame, keypoints_2_frame, frame_count):
        if not keypoints_1 or not keypoints_2:
            logger.warning("One or both keypoint inputs are empty. Returning empty keypoints.")
            return ([],)

        start_pose_data, start_width, start_height = self._get_pose_from_dataset(keypoints_1, keypoints_1_frame)
        end_pose_data, end_width, end_height = self._get_pose_from_dataset(keypoints_2, keypoints_2_frame)

        if start_pose_data is None or end_pose_data is None:
            logger.error("Could not extract valid start or end pose for interpolation.")
            raise InterruptProcessingException("Invalid start or end pose data for interpolation.")

        # Part D (Output Canvas Size Handling): Check canvas dimensions and set target_width/height
        target_width = start_width
        target_height = start_height

        if start_width != end_width or start_height != end_height:
            logger.warning(f"Canvas dimensions mismatch: Start ({start_width}x{start_height}), End ({end_width}x{end_height}). Using largest dimensions for output.")
            target_width = max(start_width, end_width)
            target_height = max(start_height, end_height)

        # Part B: Check Compatibility
        # Define all possible keypoint subsets
        keypoint_subsets = [
            "pose_keypoints_2d",
            "face_keypoints_2d",
            "hand_left_keypoints_2d",
            "hand_right_keypoints_2d",
        ]

        common_subsets = {}
        for subset_name in keypoint_subsets:
            start_kpts = start_pose_data.get(subset_name)
            end_kpts = end_pose_data.get(subset_name)

            if start_kpts is not None and end_kpts is not None:
                # Ensure they have the same number of keypoints for interpolation
                # Keypoints are stored as [x, y, conf, x, y, conf, ...]
                # So the length of the list should be a multiple of 3
                if len(start_kpts) == len(end_kpts) and len(start_kpts) % 3 == 0:
                    common_subsets[subset_name] = (np.array(start_kpts).reshape(-1, 3), np.array(end_kpts).reshape(-1, 3))
                else:
                    logger.warning(f"Subset '{subset_name}' has different number of keypoints ({len(start_kpts)} vs {len(end_kpts)}) or invalid format. Skipping interpolation for this subset.")
            elif start_kpts is not None or end_kpts is not None:
                logger.warning(f"Subset '{subset_name}' is present in one pose but not the other. Skipping interpolation for this subset.")

        if not common_subsets:
            logger.error("No common keypoint subsets found for interpolation.")
            raise InterruptProcessingException("Cannot interpolate without common keypoints.")

        # Part C: Perform Linear Interpolation
        interpolated_frames = []
        for i in range(frame_count):
            t = 0.0 if frame_count == 1 else i / (frame_count - 1)
            
            interpolated_person_data = {}
            # Copy other non-keypoint data from start_pose_data if necessary
            # For example, face_box if it exists and is not interpolated
            if "face_box" in start_pose_data and "face_box" in end_pose_data:
                start_box = np.array(start_pose_data["face_box"])
                end_box = np.array(end_pose_data["face_box"])
                interpolated_person_data["face_box"] = (start_box * (1 - t) + end_box * t).tolist()
            # ... potentially other data to copy or interpolate ...

            for subset_name, (start_kpts_np, end_kpts_np) in common_subsets.items():
                interpolated_kpts_np = (start_kpts_np * (1 - t) + end_kpts_np * t).reshape(-1).tolist()
                interpolated_person_data[subset_name] = interpolated_kpts_np
            
            interpolated_frame = {
                "people": [interpolated_person_data],
                "canvas_width": target_width,
                "canvas_height": target_height,
            }
            interpolated_frames.append(interpolated_frame)
        
        # Placeholder for Part D logic (return the interpolated_frames)
        logger.info(f"Linear interpolation completed. Generated {len(interpolated_frames)} frames.")
        
        # Temporarily returning the interpolated frames
        return (interpolated_frames,)

NODE_CLASS_MAPPINGS = {"PoseInterpolation": PoseInterpolation}

NODE_DISPLAY_NAME_MAPPINGS = {"PoseInterpolation": "DWposeDeluxe Pose Interpolation"}
