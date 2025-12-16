# ComfyUI_DWposeDeluxe/nodes/edit_nodes.py

from ..node_configs import DWposeNodeBase
from ..scripts import logger
from .keypoint_converter import detect_coordinate_format, convert_pose
import copy
import json


def ensure_list(keypoints_data):
    if isinstance(keypoints_data, str):
        try:
            keypoints_data = json.loads(keypoints_data)
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON string in keypoint data.")
            return []
    
    if isinstance(keypoints_data, dict):
        keypoints_data = [keypoints_data]
    
    if not isinstance(keypoints_data, list):
        return []
    
    return keypoints_data


class KeypointRangeFromBatch(DWposeNodeBase):
    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_keypoints",)
    FUNCTION = "execute"
    CATEGORY = "DWposeDeluxe"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoints": ("POSE_KEYPOINT",),
                "start_frame": ("INT", {"default": 0, "min": 0, "step": 1}),
                "frame_count": ("INT", {"default": 1, "min": 1, "step": 1}),
            }
        }

    def execute(self, pose_keypoints, start_frame, frame_count):
        pose_keypoints = ensure_list(pose_keypoints)
        
        if not pose_keypoints:
            return ([],)
        
        total_frames = len(pose_keypoints)
        if start_frame >= total_frames:
             logger.warning(f"KeypointRangeFromBatch: Start frame {start_frame} is beyond total frames {total_frames}. Returning empty.")
             return ([],)

        end_frame = start_frame + frame_count
        sliced = pose_keypoints[start_frame:end_frame]
        return (sliced,)

class BatchKeypoints(DWposeNodeBase):
    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_keypoints",)
    FUNCTION = "execute"
    CATEGORY = "DWposeDeluxe"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keypoints_1": ("POSE_KEYPOINT",),
                "keypoints_2": ("POSE_KEYPOINT",),
            }
        }


    def execute(self, keypoints_1, keypoints_2):
        keypoints_1 = ensure_list(keypoints_1)
        keypoints_2 = ensure_list(keypoints_2)

        if not keypoints_1 and not keypoints_2:
             return ([],)
        if not keypoints_1: return (keypoints_2,)
        if not keypoints_2: return (keypoints_1,)

        # Check dimensions
        k1_w = keypoints_1[0].get("canvas_width")
        k1_h = keypoints_1[0].get("canvas_height")
        k2_w = keypoints_2[0].get("canvas_width")
        k2_h = keypoints_2[0].get("canvas_height")

        if k1_w != k2_w or k1_h != k2_h:
            logger.warning(f"BatchKeypoints: Canvas dimensions mismatch (1: {k1_w}x{k1_h}, 2: {k2_w}x{k2_h}). Result might be inconsistent.")
        
        return (keypoints_1 + keypoints_2,)


class MergeKeypoints(DWposeNodeBase):
    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_keypoints",)
    FUNCTION = "execute"
    CATEGORY = "DWposeDeluxe"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keypoints_1": ("POSE_KEYPOINT",),
                "keypoints_2": ("POSE_KEYPOINT",),
            }
        }


    def execute(self, keypoints_1, keypoints_2):
        keypoints_1 = ensure_list(keypoints_1)
        keypoints_2 = ensure_list(keypoints_2)

        if not keypoints_1 or not keypoints_2:
            logger.warning("MergeKeypoints: One input is empty. Returning the non-empty one or empty.")
            return (keypoints_1 if keypoints_1 else keypoints_2,)
        
        if len(keypoints_1) != len(keypoints_2):
             logger.error(f"MergeKeypoints: Frame count mismatch ({len(keypoints_1)} vs {len(keypoints_2)}). Cannot merge.")
             return (keypoints_1,)

        k1_w = keypoints_1[0].get("canvas_width")
        k1_h = keypoints_1[0].get("canvas_height")
        k2_w = keypoints_2[0].get("canvas_width")
        k2_h = keypoints_2[0].get("canvas_height")

        if k1_w != k2_w or k1_h != k2_h:
             logger.warning(f"MergeKeypoints: Canvas dimensions mismatch (1: {k1_w}x{k1_h}, 2: {k2_w}x{k2_h}). Merged canvas will use dimensions from input 1.")
        
        fmt1 = detect_coordinate_format(keypoints_1)
        fmt2 = detect_coordinate_format(keypoints_2)
        
        kp2_aligned = copy.deepcopy(keypoints_2)
        
        if fmt1 != fmt2:
             logger.info(f"MergeKeypoints: Converting input 2 format ({fmt2}) to match input 1 ({fmt1}).")
             kp2_aligned = convert_pose(kp2_aligned, k1_w, k1_h, fmt2, fmt1, add_canvas_size_to_output=False)

        merged_data = []
        for i in range(len(keypoints_1)):
            frame1 = keypoints_1[i]
            frame2 = kp2_aligned[i]
            
            new_frame = copy.deepcopy(frame1)
            new_frame["people"].extend(frame2.get("people", []))
            merged_data.append(new_frame)
            
        return (merged_data,)

NODE_CLASS_MAPPINGS = {
    "KeypointRangeFromBatch": KeypointRangeFromBatch,
    "BatchKeypoints": BatchKeypoints,
    "MergeKeypoints": MergeKeypoints
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KeypointRangeFromBatch": "DWposeDeluxe Range From Batch",
    "BatchKeypoints": "DWposeDeluxe Batch Keypoints",
    "MergeKeypoints": "DWposeDeluxe Merge Keypoints"
}
