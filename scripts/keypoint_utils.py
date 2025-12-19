# ComfyUI_DWposeDeluxe/scripts/keypoint_utils.py

import json
import copy
import numpy as np
from .logger import logger

class UnifiedKeypointHandler:
    @staticmethod
    def detect_format(data, soft_boundary=1.1):
        """
        Detects if keypoints are 'normalized' or 'absolute'.
        Uses a soft boundary to allow for points slightly outside the [0, 1] range.
        """
        if not data:
            return "unknown"
        
        # Check a sample of points from the first few frames
        for frame in data[:5]:
            for person in frame.get("people", []):
                for key in ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
                    pts = person.get(key, [])
                    if pts:
                        # Check first 5 triplets (x, y, conf)
                        for i in range(0, min(len(pts), 15), 3):
                            if abs(pts[i]) > soft_boundary or abs(pts[i+1]) > soft_boundary:
                                return "absolute"
        return "normalized"

    @staticmethod
    def to_normalized(data, width, height):
        """Converts absolute keypoints to normalized (0-1)."""
        if width <= 0 or height <= 0:
            return data
            
        new_data = copy.deepcopy(data)
        for frame in new_data:
            for person in frame.get("people", []):
                for key in ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
                    if key in person:
                        pts = person[key]
                        for i in range(0, len(pts), 3):
                            pts[i] /= width
                            pts[i+1] /= height
            frame["canvas_width"] = width
            frame["canvas_height"] = height
        return new_data

    @staticmethod
    def to_absolute(data, width, height):
        """Converts normalized keypoints to absolute pixels."""
        if width <= 0 or height <= 0:
            return data

        new_data = copy.deepcopy(data)
        for frame in new_data:
            for person in frame.get("people", []):
                for key in ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
                    if key in person:
                        pts = person[key]
                        for i in range(0, len(pts), 3):
                            pts[i] *= width
                            pts[i+1] *= height
            frame["canvas_width"] = width
            frame["canvas_height"] = height
        return new_data

    @staticmethod
    def reset_confidence(data, threshold=0.25):
        """Sets confidence to 1.0 if >= threshold, else 0.0. For external compatibility."""
        new_data = copy.deepcopy(data)
        for frame in new_data:
            for person in frame.get("people", []):
                for key in ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
                    if key in person:
                        pts = person[key]
                        for i in range(2, len(pts), 3):
                            pts[i] = 1.0 if pts[i] >= threshold else 0.0
        return new_data
