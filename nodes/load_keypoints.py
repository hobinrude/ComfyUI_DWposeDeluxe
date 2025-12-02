import os
import json
import folder_paths

def detect_format(data):
    for item in data:
        for person in item.get("people", []):
            pts = person.get("pose_keypoints_2d", [])
            if pts:
                xs = pts[0::3][:5]
                ys = pts[1::3][:5]
                if any(x > 2 or y > 2 for x, y in zip(xs, ys)):
                    return "absolute"
                else:
                    return "normalized"
    return "unknown"

def detect_structure(json_string: str) -> str:
    if ',\n' in json_string:
        return "pretty"
    return "default"

def get_keypoint_files_from_input_dir():
    input_dir = folder_paths.get_input_directory()
    files = []
    for f in os.listdir(input_dir):
        if f.lower().endswith(".json") and os.path.isfile(os.path.join(input_dir, f)):
            files.append(f)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(input_dir, x)), reverse=True)
    if not files:
        files = ["[none]"]
    return files


class LoadPoseKeypoints:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {

                "file": (get_keypoint_files_from_input_dir(), {
                    "dwpose_adv.file_select_binding": True
                }),
            },

        }

    CATEGORY = "DWPose Advanced/IO"

    RETURN_TYPES = ("POSE_KEYPOINT", "STRING")
    RETURN_NAMES = ("pose_keypoints", "keypoint_info")
    FUNCTION = "load_keypoints"

    def load_keypoints(self, file: str):
        if file == "[none]" or not file or not file.strip():
            return ("", "No file selected.")
        input_dir = folder_paths.get_input_directory()
        file_path_to_load = os.path.join(input_dir, file)

        if not os.path.exists(file_path_to_load):
            return ("", f"Error: File not found at {file_path_to_load}")

        with open(file_path_to_load, 'r', encoding='utf-8') as f:
            data_str = f.read()
        
        try:
            data = json.loads(data_str)
            
            if isinstance(data, dict):
                data = [data] # Wrap single dictionary in a list

            if not data:
                return ("", "Error: JSON file is empty or malformed.")

            pose_format = detect_format(data)
            json_structure = detect_structure(data_str)

            confidence_variable = False
            for frame in data:
                for person in frame.get("people", []):
                    for key in ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
                        pts = person.get(key)
                        if pts:
                            for i in range(2, len(pts), 3):
                                if pts[i] not in [0.0, 1.0, 2.0]:
                                    confidence_variable = True
                                    break
                        if confidence_variable:
                            break
                if confidence_variable:
                    break
            
            expected_counts = {
                "body": 18, "feet": 6, "face": 68,
                "left_hand": 21, "right_hand": 21
            }
            present_counts = {k: 0 for k in expected_counts}
            total_expected = {k: 0 for k in expected_counts}
            subset_exists = {k: False for k in expected_counts}

            for frame in data:
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
                    
                    subset_map = {
                        "face": "face_keypoints_2d",
                        "left_hand": "hand_left_keypoints_2d",
                        "right_hand": "hand_right_keypoints_2d"
                    }
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

            frame_count = len(data)
            poses_per_frame = [len(frame.get("people", [])) for frame in data if isinstance(frame, dict)]
            number_of_people = max(poses_per_frame) if poses_per_frame else 0

            first_frame_data = data[0] if data else {}
            width = first_frame_data.get("canvas_width", "n/a")
            height = first_frame_data.get("canvas_height", "n/a")

            keypoint_info = (
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

        except json.JSONDecodeError:
            keypoint_info = "Error: Invalid JSON file."
            data = None
        except Exception as e:
            keypoint_info = f"Error processing file: {e}"
            data = None

        return (data_str if data is not None else "", keypoint_info)

    @classmethod
    def IS_CHANGED(s, file, **kwargs):
        if file == "[none]" or not file or not file.strip():
            return float("nan")
        input_dir = folder_paths.get_input_directory()
        file_path_to_load = os.path.join(input_dir, file)
        
        try:
            return os.path.getmtime(file_path_to_load)
        except:
            return float("nan")

NODE_CLASS_MAPPINGS = {
    "LoadPoseKeypoints": LoadPoseKeypoints
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadPoseKeypoints": "DWposeDeluxe Load Keypoints"
}