# ComfyUI_DWposeDeluxe/nodes/keypoint_cherry_picker.py

from ..node_configs import DWposeNodeBase
from ..scripts import logger
import json
import copy

BODY_POINT_NAMES = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
    "RAnkle", "LHip", "LKnee", "LAnkle", "REye",
    "LEye", "REar", "LEar"
]

def ensure_list(keypoints_data):
    if isinstance(keypoints_data, str):
        try:
            return json.loads(keypoints_data)
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON string in keypoint data.")
            return []
    if isinstance(keypoints_data, dict):
        return [keypoints_data]
    if not isinstance(keypoints_data, list):
        return []
    return keypoints_data

def log_cherry_pick_summary(node_name, gains_per_point, num_frames):
    logger.info(f"[{node_name}] Confidence Gain Summary (Average per frame):")
    total_avg_gain = 0.0
    divisor = max(1, num_frames)
    
    for i, name in enumerate(BODY_POINT_NAMES):
        avg_gain = gains_per_point.get(i, 0.0) / divisor
        logger.info(f"  - {name:10}: {avg_gain:+.4f}")
        total_avg_gain += avg_gain
    
    overall_avg = total_avg_gain / len(BODY_POINT_NAMES)
    logger.info(f"[{node_name}] Overall Average Confidence Gain: {overall_avg:+.4f}")

class CherryPickerFrames(DWposeNodeBase):
    CATEGORY = "DWposeDeluxe"
    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_keypoints",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoints": ("POSE_KEYPOINT",),
                "factor": ("INT", {"default": 2, "min": 2, "max": 100, "step": 1}),
            }
        }

    def execute(self, pose_keypoints, factor):
        data = ensure_list(pose_keypoints)
        if not data:
            return ([],)

        data = copy.deepcopy(data)
        result_data = []
        num_frames = len(data)

        gains_per_point = {i: 0.0 for i in range(len(BODY_POINT_NAMES))}
        output_frame_count = 0

        # Iterate in chunks
        for i in range(0, num_frames, factor):
            chunk = data[i : i + factor]
            if not chunk:
                continue

            merged_frame = copy.deepcopy(chunk[0])
            merged_people = []

            max_people = 0
            for frame in chunk:
                max_people = max(max_people, len(frame.get("people", [])))

            for p_idx in range(max_people):
                candidate_person_entries = []
                for frame in chunk:
                    people = frame.get("people", [])
                    candidate_person_entries.append(people[p_idx] if p_idx < len(people) else None)

                if all(c is None for c in candidate_person_entries):
                    continue

                first_valid = next(item for item in candidate_person_entries if item is not None)
                merged_person = copy.deepcopy(first_valid)
                
                kp_types = ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]
                
                for kp_type in kp_types:
                    arrays_to_merge = []
                    for cp in candidate_person_entries:
                        arrays_to_merge.append(cp[kp_type] if cp and kp_type in cp else None)
                    
                    if any(arr is not None for arr in arrays_to_merge):
                        valid_arr = next(arr for arr in arrays_to_merge if arr is not None)
                        length = len(valid_arr)
                        merged_arr = [0.0] * length

                        for k_start in range(0, length, 3):
                            best_score = -1.0
                            best_triplet = (0.0, 0.0, 0.0)
                            found_any = False
                            all_scores_in_chunk = []

                            for arr in arrays_to_merge:
                                if arr and len(arr) > k_start + 2:
                                    score = arr[k_start + 2]
                                    all_scores_in_chunk.append(score)
                                    if score > best_score:
                                        best_score = score
                                        best_triplet = (arr[k_start], arr[k_start+1], score)
                                        found_any = True
                                else:
                                    all_scores_in_chunk.append(0.0)
                            
                            if found_any:
                                merged_arr[k_start] = best_triplet[0]
                                merged_arr[k_start+1] = best_triplet[1]
                                merged_arr[k_start+2] = best_triplet[2]

                                # Tracking gain for body points
                                if kp_type == "pose_keypoints_2d":
                                    point_idx = k_start // 3
                                    if point_idx < len(BODY_POINT_NAMES):
                                        # Gain = best - average(others)
                                        if len(all_scores_in_chunk) > 1:
                                            # Find index of the best score to remove it from the average
                                            # We use the first occurrence to be safe
                                            best_idx = all_scores_in_chunk.index(best_score)
                                            others = all_scores_in_chunk[:best_idx] + all_scores_in_chunk[best_idx+1:]
                                            avg_others = sum(others) / len(others)
                                            gain = best_score - avg_others
                                        else:
                                            gain = 0.0
                                        gains_per_point[point_idx] += gain
                        
                        merged_person[kp_type] = merged_arr

                merged_people.append(merged_person)

            merged_frame["people"] = merged_people
            result_data.append(merged_frame)
            output_frame_count += 1

        log_cherry_pick_summary("CherryPickerFrames", gains_per_point, output_frame_count)
        return (result_data,)

class CherryPickerTwoInputs(DWposeNodeBase):
    CATEGORY = "DWposeDeluxe"
    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_keypoints",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keypoints_1": ("POSE_KEYPOINT",),
                "keypoints_2": ("POSE_KEYPOINT",),
            }
        }

    def execute(self, keypoints_1, keypoints_2):
        data1 = ensure_list(keypoints_1)
        data2 = ensure_list(keypoints_2)

        if not data1: return (data2,)
        if not data2: return (data1,)

        if len(data1) != len(data2):
            logger.error(f"CherryPickerTwoInputs: Frame count mismatch ({len(data1)} vs {len(data2)}).")
            return (data1,)

        data1 = copy.deepcopy(data1)
        data2 = copy.deepcopy(data2)
        result_data = []
        num_frames = len(data1)

        gains_per_point = {i: 0.0 for i in range(len(BODY_POINT_NAMES))}

        for i in range(num_frames):
            frame1 = data1[i]
            frame2 = data2[i]

            merged_frame = copy.deepcopy(frame1)
            merged_people = []

            people1 = frame1.get("people", [])
            people2 = frame2.get("people", [])
            max_people = max(len(people1), len(people2))

            for p_idx in range(max_people):
                p1 = people1[p_idx] if p_idx < len(people1) else None
                p2 = people2[p_idx] if p_idx < len(people2) else None

                if p1 is None and p2 is None:
                    continue
                if p1 is None:
                    # Input 2 has person, Input 1 doesn't. Gain is Input 2 scores.
                    p2_copy = copy.deepcopy(p2)
                    merged_people.append(p2_copy)
                    if "pose_keypoints_2d" in p2_copy:
                        arr = p2_copy["pose_keypoints_2d"]
                        for k_start in range(0, len(arr), 3):
                            point_idx = k_start // 3
                            if point_idx < len(BODY_POINT_NAMES):
                                gains_per_point[point_idx] += arr[k_start + 2]
                    continue
                if p2 is None:
                    # Input 1 has person, Input 2 doesn't. Gain is Input 1 scores.
                    p1_copy = copy.deepcopy(p1)
                    merged_people.append(p1_copy)
                    if "pose_keypoints_2d" in p1_copy:
                        arr = p1_copy["pose_keypoints_2d"]
                        for k_start in range(0, len(arr), 3):
                            point_idx = k_start // 3
                            if point_idx < len(BODY_POINT_NAMES):
                                gains_per_point[point_idx] += arr[k_start + 2]
                    continue

                # Both present, merge best points
                merged_person = copy.deepcopy(p1)
                kp_types = ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]
                
                for kp_type in kp_types:
                    arr1 = p1.get(kp_type)
                    arr2 = p2.get(kp_type)
                    
                    if arr1 is not None and arr2 is not None:
                        length = min(len(arr1), len(arr2))
                        merged_arr = list(arr1) # Baseline for structure
                        
                        for k_start in range(0, length, 3):
                            if k_start + 2 < len(arr2):
                                score1 = arr1[k_start + 2]
                                score2 = arr2[k_start + 2]
                                
                                # Always pick the better one
                                if score2 > score1:
                                    merged_arr[k_start] = arr2[k_start]
                                    merged_arr[k_start+1] = arr2[k_start+1]
                                    merged_arr[k_start+2] = score2
                                
                                # Gain = abs(A - B) representing the "competitive gap"
                                if kp_type == "pose_keypoints_2d":
                                    point_idx = k_start // 3
                                    if point_idx < len(BODY_POINT_NAMES):
                                        gains_per_point[point_idx] += abs(score1 - score2)
                        
                        merged_person[kp_type] = merged_arr
                    elif arr2 is not None:
                        merged_person[kp_type] = copy.deepcopy(arr2)
                        if kp_type == "pose_keypoints_2d":
                            for k_start in range(0, len(arr2), 3):
                                point_idx = k_start // 3
                                if point_idx < len(BODY_POINT_NAMES):
                                    gains_per_point[point_idx] += arr2[k_start + 2]
                    elif arr1 is not None:
                         if kp_type == "pose_keypoints_2d":
                            for k_start in range(0, len(arr1), 3):
                                point_idx = k_start // 3
                                if point_idx < len(BODY_POINT_NAMES):
                                    gains_per_point[point_idx] += arr1[k_start + 2]

                merged_people.append(merged_person)

            merged_frame["people"] = merged_people
            result_data.append(merged_frame)

        log_cherry_pick_summary("CherryPickerTwoInputs", gains_per_point, num_frames)
        return (result_data,)

NODE_CLASS_MAPPINGS = {
    "CherryPickerFrames": CherryPickerFrames,
    "CherryPickerTwoInputs": CherryPickerTwoInputs
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CherryPickerFrames": "DWposeDeluxe Cherry Picker (Frames)",
    "CherryPickerTwoInputs": "DWposeDeluxe Cherry Picker (2 Inputs)"
}
