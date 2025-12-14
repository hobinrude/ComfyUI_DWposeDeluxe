from ..node_configs import DWposeNodeBase
import numpy as np
import torch
import cv2
import random
import json
import copy
from ..scripts import logger

class PoseTrackerNode(DWposeNodeBase):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "keypoints": ("POSE_KEYPOINT",),
                "center_method": (["bbox_centroid", "gravity_center"],),
            },
            "optional": {
                "distance_threshold": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 1000.0, "step": 1.0}),
                "max_frame_age": ("INT", {"default": 24, "min": 1, "max": 1000, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "track_poses"
    CATEGORY = "DWposeDeluxe"

    def calculate_bbox_centroid(self, person):
        flat_keypoints = []
        # Aggregate all available keypoints to derive a true full-pose bounding box
        # According to xframes_vs_xposes.md, 'pose_keypoints_2d' is the primary one.
        for key in ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
            if key in person and person[key] is not None:
                flat_keypoints.extend(person[key])

        if not flat_keypoints:
            logger.warning("No keypoints found for person to calculate bbox_centroid.")
            return None

        valid_xs = []
        valid_ys = []
        for i in range(0, len(flat_keypoints), 3):
            if i + 2 < len(flat_keypoints): # Ensure triplet is complete
                x, y, confidence = flat_keypoints[i], flat_keypoints[i+1], flat_keypoints[i+2]
                if confidence > 0:
                    valid_xs.append(x)
                    valid_ys.append(y)
        
        if not valid_xs:
            logger.warning("No valid keypoints (confidence > 0) found for person to calculate bbox_centroid.")
            return None
        
        x1, x2 = min(valid_xs), max(valid_xs)
        y1, y2 = min(valid_ys), max(valid_ys)

        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def calculate_gravity_center(self, person):
        flat_keypoints = []
        # Aggregate all available keypoints
        for key in ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
            if key in person and person[key] is not None:
                flat_keypoints.extend(person[key])
        
        if not flat_keypoints:
            logger.warning("No keypoints found for person to calculate gravity_center.")
            return None

        valid_points = []
        # The list is flat [x1, y1, c1, x2, y2, c2, ...], so iterate in steps of 3
        for i in range(0, len(flat_keypoints), 3):
            # Ensure there are enough elements for a full keypoint triplet
            if i + 2 < len(flat_keypoints):
                x, y, confidence = flat_keypoints[i], flat_keypoints[i+1], flat_keypoints[i+2]
                if confidence > 0:
                    valid_points.append((x, y)) # Store as tuples/points

        if not valid_points:
            logger.warning("No valid keypoints (confidence > 0) found for person to calculate gravity_center.")
            return None

        # Calculate the average position
        num_valid_points = len(valid_points)
        avg_x = sum(p[0] for p in valid_points) / num_valid_points
        avg_y = sum(p[1] for p in valid_points) / num_valid_points
        
        return (int(avg_x), int(avg_y))

    def track_poses(self, keypoints, center_method, distance_threshold=50.0, max_frame_age=24):
        if isinstance(keypoints, str):
            try:
                keypoints = json.loads(keypoints)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse keypoints JSON string: {e}")
                return (torch.zeros((1, 512, 512, 3), dtype=torch.float32),)
        else:
            keypoints = copy.deepcopy(keypoints)

        if not keypoints:
            # Return a blank image if no keypoints are provided
            return (torch.zeros((1, 512, 512, 3), dtype=torch.float32),)

        canvas_height = keypoints[0].get('canvas_height')
        canvas_width = keypoints[0].get('canvas_width')

        # Fallback to estimating canvas size if not explicitly provided
        if not canvas_height or not canvas_width:
             logger.warning("canvas_width or canvas_height not found in keypoint data. Attempting to estimate from all keypoints.")
             all_xs = []
             all_ys = []
             for frame_data in keypoints:
                 for person in frame_data.get('people', []):
                     flat_keypoints = []
                     for key in ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
                         if key in person and person[key] is not None:
                             flat_keypoints.extend(person[key])
                     for i in range(0, len(flat_keypoints), 3):
                         if i + 2 < len(flat_keypoints):
                             x, y, confidence = flat_keypoints[i], flat_keypoints[i+1], flat_keypoints[i+2]
                             if confidence > 0:
                                 all_xs.append(x)
                                 all_ys.append(y)
             if all_xs and all_ys:
                canvas_width = int(max(all_xs))
                canvas_height = int(max(all_ys))
                logger.info(f"Estimated canvas size to be {canvas_width}x{canvas_height}.")
             else:
                logger.error("Could not determine canvas size from keypoints. Falling back to 512x512.")
                canvas_height = 512
                canvas_width = 512
        
        # Calculate decay factor based on max_frame_age
        raw_fade_duration = max(1, max_frame_age / 2)
        raw_decay = 1.0 / raw_fade_duration

        output_frames = []
        markers_to_draw = [] 

        for frame_data in keypoints:
            frame_canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
            
            # Step 1: Decay opacity of all existing markers and filter out faded ones
            next_markers = []
            for marker in markers_to_draw:
                marker['opacity'] -= raw_decay
                if marker['opacity'] > 0:
                    next_markers.append(marker)
            markers_to_draw = next_markers

            # Step 2: Add new markers from the current frame
            people = frame_data.get('people', [])
            for person in people:
                centroid = None
                if center_method == 'bbox_centroid':
                    centroid = self.calculate_bbox_centroid(person)
                elif center_method == 'gravity_center':
                    centroid = self.calculate_gravity_center(person)

                if centroid and (0 <= centroid[0] < canvas_width and 0 <= centroid[1] < canvas_height):
                    markers_to_draw.append({'pos': centroid, 'opacity': 1.0})
            
            # Step 3: Draw all markers on the canvas
            # Sort by opacity to draw fainter ones first, though with solid circles it won't have a visual difference
            for marker in sorted(markers_to_draw, key=lambda m: m['opacity']):
                opacity = marker['opacity']
                if opacity > 1.0: opacity = 1.0 # Ensure opacity doesn't exceed 1
                
                color_intensity = int(255 * opacity)
                cv2.circle(frame_canvas, marker['pos'], 24, (color_intensity, color_intensity, color_intensity), -1)

            output_frames.append(frame_canvas)

        if not output_frames:
             return (torch.zeros((1, canvas_height, canvas_width, 3), dtype=torch.float32),)

        # Convert list of numpy arrays to a single torch tensor
        video_tensor = torch.from_numpy(np.array(output_frames).astype(np.float32) / 255.0)

        return (video_tensor,)


NODE_CLASS_MAPPINGS = {"PoseTrackerNode": PoseTrackerNode}

NODE_DISPLAY_NAME_MAPPINGS = {"PoseTrackerNode": "DWposeDeluxe Tracker Node"}