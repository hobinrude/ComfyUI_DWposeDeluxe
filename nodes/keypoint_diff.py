# ComfyUI_DWposeDeluxe/nodes/keypoint_diff.py

import json
import copy
import math
import numpy as np
from ..scripts.logger import logger
from ..scripts.keypoint_utils import UnifiedKeypointHandler
from ..node_configs import DWposeNodeBase

class KeypointDiff(DWposeNodeBase):
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("diff_info",)
    FUNCTION = "execute"
    CATEGORY = "DWposeDeluxe"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keypoints_1": ("POSE_KEYPOINT",),
                "keypoints_2": ("POSE_KEYPOINT",),
                "coordinate_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "confidence_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    def execute(self, keypoints_1, keypoints_2, coordinate_threshold, confidence_threshold):
        if not keypoints_1 or not keypoints_2:
            return ("Error: One or both keypoint inputs are empty.",)

        # Ensure we are working with lists
        if isinstance(keypoints_1, str): keypoints_1 = json.loads(keypoints_1)
        if isinstance(keypoints_2, str): keypoints_2 = json.loads(keypoints_2)

        len1 = len(keypoints_1)
        len2 = len(keypoints_2)

        report = []
        report.append(f"--- Keypoint Diff Report ---")
        report.append(f"Batch 1 Frames: {len1}")
        report.append(f"Batch 2 Frames: {len2}")

        if len1 != len2:
            msg = f"Frame count mismatch: {len1} vs {len2}"
            logger.warning(f"[KeypointDiff] {msg}")
            report.append(msg)
        
        num_frames = min(len1, len2)
        total_triplets = 0
        
        # Categorized tracking
        categories = {
            "pose_keypoints_2d": {"total": 0, "coord_mismatch": 0, "conf_mismatch": 0, "sum_dist": 0.0, "sum_conf_diff": 0.0},
            "face_keypoints_2d": {"total": 0, "coord_mismatch": 0, "conf_mismatch": 0, "sum_dist": 0.0, "sum_conf_diff": 0.0},
            "hand_left_keypoints_2d": {"total": 0, "coord_mismatch": 0, "conf_mismatch": 0, "sum_dist": 0.0, "sum_conf_diff": 0.0},
            "hand_right_keypoints_2d": {"total": 0, "coord_mismatch": 0, "conf_mismatch": 0, "sum_dist": 0.0, "sum_conf_diff": 0.0},
        }

        # Detect format for percentage calculation
        kp_format = UnifiedKeypointHandler.detect_format(keypoints_1)
        canvas_w = keypoints_1[0].get("canvas_width", 512)
        canvas_h = keypoints_1[0].get("canvas_height", 512)
        max_dim = max(canvas_w, canvas_h)
        
        # Scale threshold for comparison if absolute
        scaled_coord_thresh = coordinate_threshold * (max_dim if kp_format == "absolute" else 1.0)

        for f_idx in range(num_frames):
            frame1 = keypoints_1[f_idx]
            frame2 = keypoints_2[f_idx]
            
            people1 = frame1.get("people", [])
            people2 = frame2.get("people", [])
            
            if len(people1) != len(people2):
                report.append(f"Frame {f_idx}: People count mismatch ({len(people1)} vs {len(people2)})")
            
            num_people = min(len(people1), len(people2))
            for p_idx in range(num_people):
                p1 = people1[p_idx]
                p2 = people2[p_idx]
                
                for key in categories.keys():
                    pts1 = p1.get(key, [])
                    pts2 = p2.get(key, [])
                    
                    if len(pts1) != len(pts2):
                        report.append(f"Frame {f_idx}, Person {p_idx}, {key}: Point count mismatch ({len(pts1)} vs {len(pts2)})")
                        continue
                    
                    # Process as triplets (x, y, conf)
                    for i in range(0, len(pts1), 3):
                        x1, y1, c1 = pts1[i:i+3]
                        x2, y2, c2 = pts2[i:i+3]
                        
                        categories[key]["total"] += 1
                        total_triplets += 1
                        
                        # Distance for coordinates
                        dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        conf_diff = abs(c1 - c2)
                        
                        categories[key]["sum_dist"] += dist
                        categories[key]["sum_conf_diff"] += conf_diff
                        
                        if dist > scaled_coord_thresh:
                            categories[key]["coord_mismatch"] += 1
                        if conf_diff > confidence_threshold:
                            categories[key]["conf_mismatch"] += 1

        if total_triplets == 0:
            report.append("No common keypoints found to compare.")
        else:
            global_coord_mismatch = sum(v["coord_mismatch"] for v in categories.values())
            global_conf_mismatch = sum(v["conf_mismatch"] for v in categories.values())
            
            report.append(f"--- Global Stats ---")
            report.append(f"Coord Mismatch: {(global_coord_mismatch / total_triplets) * 100:.4f}% ({global_coord_mismatch}/{total_triplets} keypoints)")
            report.append(f"Conf Mismatch:  {(global_conf_mismatch / total_triplets) * 100:.4f}% ({global_conf_mismatch}/{total_triplets} keypoints)")
            
            # Calculate Average Value Mismatch (%)
            # For coordinates, if normalized, avg_dist * 100 is the percentage. 
            # If absolute, we normalize by canvas max dimension.
            total_sum_dist = sum(v["sum_dist"] for v in categories.values())
            total_sum_conf = sum(v["sum_conf_diff"] for v in categories.values())
            
            avg_dist = total_sum_dist / total_triplets
            avg_conf = total_sum_conf / total_triplets
            
            if kp_format == "absolute":
                avg_coord_pct = (avg_dist / max_dim) * 100
            else:
                avg_coord_pct = avg_dist * 100
            
            avg_conf_pct = avg_conf * 100 # Confidence is always 0..1
            
            report.append(f"Avg Coord Value Mismatch: {avg_coord_pct:.6f}%")
            report.append(f"Avg Conf Value Mismatch:  {avg_conf_pct:.6f}%")
            
            report.append(f"--- Categorized Mismatch ---")
            for key, stats in categories.items():
                if stats["total"] > 0:
                    c_pct = (stats["coord_mismatch"] / stats["total"]) * 100
                    f_pct = (stats["conf_mismatch"] / stats["total"]) * 100
                    report.append(f" - {key:23}: Coord {c_pct:8.4f}% | Conf {f_pct:8.4f}%")

        final_report = "\n".join(report)
        
        if total_triplets > 0 and (global_coord_mismatch > 0 or global_conf_mismatch > 0 or len1 != len2):
            logger.warning(f"[KeypointDiff] Discrepancies detected!\n{final_report}")
        else:
            logger.info(f"[KeypointDiff] Perfect match! ({total_triplets} keypoints compared across {num_frames} frames)")

        return (final_report,)

NODE_CLASS_MAPPINGS = {"KeypointDiff": KeypointDiff}
NODE_DISPLAY_NAME_MAPPINGS = {"KeypointDiff": "DWpose Keypoint Diff"}