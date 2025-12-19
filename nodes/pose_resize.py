# ComfyUI_DWposeDeluxe/nodes/pose_resize.py

from ..node_configs import DWposeNodeBase
from ..scripts.keypoint_utils import UnifiedKeypointHandler
import json
import copy
import numpy as np

from comfy.model_management import InterruptProcessingException
from ..scripts import logger

class PoseResize(DWposeNodeBase):
    CATEGORY = "DWposeDeluxe"
    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_keypoints",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoints": ("POSE_KEYPOINT",),
                "crop_to_pose": ("BOOLEAN", {"default": False}),
                "width": ("INT", {"default": 0, "min": 0, "step": 1}),
                "height": ("INT", {"default": 0, "min": 0, "step": 1}),
                "resize_method": (["stretch", "pad/crop"],),
                "left": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
                "right": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
                "top": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
                "bottom": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
                "extra_padding": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
                "x_offset": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
                "y_offset": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
                "divisible_by": ("INT", {"default": 2, "min": 1, "max": 64, "step": 1}),
            }
        }

    def execute(self, pose_keypoints, crop_to_pose, width, height, resize_method, left, right, top, bottom, extra_padding, x_offset, y_offset, divisible_by):
        if not pose_keypoints:
            return ([],)

        # 1. Initial Setup: Load data and convert to absolute coordinates
        data = json.loads(pose_keypoints) if isinstance(pose_keypoints, str) else copy.deepcopy(pose_keypoints)
        if not data:
            return ([],)

        input_format = UnifiedKeypointHandler.detect_format(data)
        
        # Capture original dimensions for conversion if normalized
        first_frame = data[0]
        orig_canvas_w = first_frame.get("canvas_width")
        orig_canvas_h = first_frame.get("canvas_height")
        
        if input_format == "normalized":
            if not orig_canvas_w or not orig_canvas_h:
                logger.error("[PoseResize] Input is normalized but 'canvas_width'/'canvas_height' are missing.")
                raise InterruptProcessingException()
            data = UnifiedKeypointHandler.to_absolute(data, orig_canvas_w, orig_canvas_h)
        
        # Initialize transformation variables
        scale_x, scale_y = 1.0, 1.0
        offset_x, offset_y = 0.0, 0.0
        
        # We now work in absolute pixels
        canvas_w = data[0].get("canvas_width", 0)
        canvas_h = data[0].get("canvas_height", 0)
        
        if canvas_w == 0 or canvas_h == 0:
            # Fallback for missing canvas size: derive from keypoints
            max_x, max_y = 0, 0
            for frame in data:
                 for person in frame.get("people", []):
                    for key in ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
                        if key in person:
                            pts = person[key]
                            for i in range(0, len(pts), 3):
                                max_x = max(max_x, pts[i])
                                max_y = max(max_y, pts[i+1])
            canvas_w, canvas_h = max_x, max_y
            if canvas_w == 0 or canvas_h == 0: return ([],)

        # 2. Crop to Pose
        if crop_to_pose:
            min_kx, min_ky = float('inf'), float('inf')
            max_kx, max_ky = float('-inf'), float('-inf')
            has_keypoints = False
            for frame in data:
                for person in frame.get("people", []):
                    for key in ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
                        if key in person:
                            pts = person[key]
                            for i in range(0, len(pts), 3):
                                if len(pts) > i + 2 and pts[i+2] > 0:
                                    has_keypoints = True
                                    min_kx, min_ky = min(min_kx, pts[i]), min(min_ky, pts[i+1])
                                    max_kx, max_ky = max(max_kx, pts[i]), max(max_ky, pts[i+1])
            if has_keypoints:
                offset_x -= min_kx
                offset_y -= min_ky
                canvas_w, canvas_h = max_kx - min_kx, max_ky - min_ky
        
        # 3. Width/Height Resize
        original_aspect_ratio = canvas_w / canvas_h if canvas_h > 0 else 1.0

        target_w = width if width > 0 else canvas_w
        target_h = height if height > 0 else canvas_h

        if width > 0 and height <= 0:
            target_h = width / original_aspect_ratio
        elif height > 0 and width <= 0:
            target_w = height * original_aspect_ratio
        
        if resize_method == "stretch":
            s_x = target_w / canvas_w if canvas_w > 0 else 1.0
            s_y = target_h / canvas_h if canvas_h > 0 else 1.0
            
            scale_x *= s_x
            scale_y *= s_y
            canvas_w *= s_x
            canvas_h *= s_y

        elif resize_method == "pad/crop":
            offset_x += (target_w - canvas_w) / 2
            offset_y += (target_h - canvas_h) / 2
            canvas_w, canvas_h = target_w, target_h

        # 4. Left/Right/Top/Bottom Padding/Cropping
        offset_x += left
        offset_y += top
        canvas_w += left + right
        canvas_h += top + bottom

        # 5. Extra Padding
        offset_x += extra_padding
        offset_y += extra_padding
        canvas_w += 2 * extra_padding
        canvas_h += 2 * extra_padding

        # 6. X/Y Offset
        offset_x += x_offset
        offset_y += y_offset

        if canvas_w <= 0 or canvas_h <= 0:
            logger.error(f"[PoseResize] Cropping overdose. Got canvas W:{canvas_w}, H:{canvas_h}")
            raise InterruptProcessingException()

        # 7. Divisible By
        if divisible_by > 1:
            rem_w = canvas_w % divisible_by
            if rem_w != 0:
                pad_w = divisible_by - rem_w
                canvas_w += pad_w
                offset_x += pad_w / 2
            rem_h = canvas_h % divisible_by
            if rem_h != 0:
                pad_h = divisible_by - rem_h
                canvas_h += pad_h
                offset_y += pad_h / 2

        # 8. Apply accumulated transformations to all keypoints
        final_canvas_w = int(canvas_w)
        final_canvas_h = int(canvas_h)
        
        for frame in data:
            for person in frame.get("people", []):
                for key in ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
                    if key in person:
                        pts = person[key]
                        for i in range(0, len(pts), 3):
                            if len(pts) > i + 2 and pts[i+2] > 0:
                                pts[i] = (pts[i] * scale_x) + offset_x
                                pts[i+1] = (pts[i+1] * scale_y) + offset_y
            
            frame["canvas_width"] = final_canvas_w
            frame["canvas_height"] = final_canvas_h
            
        # 9. Convert back to normalized if original was normalized
        if input_format == "normalized" and final_canvas_w > 0 and final_canvas_h > 0:
            data = UnifiedKeypointHandler.to_normalized(data, final_canvas_w, final_canvas_h)

        return (data,)

NODE_CLASS_MAPPINGS = {"PoseResize": PoseResize}
NODE_DISPLAY_NAME_MAPPINGS = {"PoseResize": "DWposeDeluxe PoseResize"}