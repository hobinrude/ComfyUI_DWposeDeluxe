# ComfyUI_DWposeDeluxe/nodes/keypoint_converter.py

import json
import os
import folder_paths
import copy

def detect_coordinate_format(data):
    for item in data:
        for person in item.get("people", []):
            pts = person.get("pose_keypoints_2d", [])
            if pts:
                for i in range(0, min(len(pts), 15), 3):
                    if abs(pts[i]) > 1.0 or abs(pts[i + 1]) > 1.0:
                        return "absolute"
    return "normalized"

def convert_pose(data, width, height, current_format, target_format, add_canvas_size_to_output=True):
    if current_format != target_format:
        for item in data:
            for person in item.get("people", []):
                for key in [
                    "pose_keypoints_2d",
                    "face_keypoints_2d",
                    "hand_left_keypoints_2d",
                    "hand_right_keypoints_2d",
                ]:
                    if key in person:
                        pts = person[key]
                        for i in range(0, len(pts), 3):
                            if target_format == "normalized":
                                if width > 0: pts[i] /= width
                                if height > 0: pts[i+1] /= height
                            else:
                                pts[i] *= width
                                pts[i+1] *= height
                        person[key] = pts
    
    if add_canvas_size_to_output:
        for item in data:
            item["canvas_width"] = width
            item["canvas_height"] = height
            
    return data


def pretty_json_triplets_converter(data):

    def format_keypoint_list_converter(kpt_list):
        if not kpt_list:
            return ""
        lines = []
        for i in range(0, len(kpt_list), 3):
            chunk = kpt_list[i:i+3]
            lines.append("                    " + ", ".join(map(str, chunk)) + ",")
        if lines:
            lines[-1] = lines[-1][:-1]
        return "\n".join(lines)
    output_frames = []
    for frame in data:
        frame_parts = []
        if "people" in frame:
            people_list_str = []
            for person in frame["people"]:
                person_parts = []
                for key, value in person.items():
                    if key.endswith("_keypoints_2d") and isinstance(value, list):
                        formatted_kpts = format_keypoint_list_converter(value)
                        person_parts.append(f'                \"{key}\": [\n{formatted_kpts}\n                ]')
                    else:
                        person_parts.append(f'                \"{key}\": {json.dumps(value)}')
                person_str = "            {\n" + ",\n".join(person_parts) + "\n            }"
                people_list_str.append(person_str)
            people_block = '        "people": [\n' + ",\n".join(people_list_str) + '\n        ]'
            frame_parts.append(people_block)
        other_keys = [k for k in frame.keys() if k != "people"]
        for key in other_keys:
            frame_parts.append(f'        \"{key}\": {json.dumps(frame[key])}')
        output_frames.append("    {\n" + ",\n".join(frame_parts) + "\n    }")
    return "[\n" + ",\n".join(output_frames) + "\n]"

def remove_empty_keypoints(data):
    cleaned_data = []
    for frame in data:
        new_frame = {}
        for key, value in frame.items():
            if key == "people":
                cleaned_people = []
                for person in value:
                    cleaned_person = {k: v for k, v in person.items() if not (isinstance(v, list) and not v)}
                    if cleaned_person:
                        cleaned_people.append(cleaned_person)
                if cleaned_people:
                    new_frame["people"] = cleaned_people
            else:
                new_frame[key] = value
        cleaned_data.append(new_frame)
    return cleaned_data


def filter_keypoints_by_confidence(data, confidence_threshold, reset_confidence=False):
    for item in data:
        for person in item.get("people", []):
            for key in [
                "pose_keypoints_2d",
                "face_keypoints_2d",
                "hand_left_keypoints_2d",
                "hand_right_keypoints_2d",
            ]:
                if key in person:
                    pts = person[key]
                    for i in range(0, len(pts), 3):
                        if reset_confidence:
                            if pts[i + 2] < confidence_threshold:
                                pts[i + 2] = 0.0
                            else:
                                pts[i + 2] = 1.0
                        else:
                            if pts[i + 2] < confidence_threshold:
                                pts[i] *= -1
                                pts[i + 1] *= -1
    return data


class KeypointConverter:
    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_keypoints",)
    FUNCTION = "execute"
    CATEGORY = "DWPose Advanced/Utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoints": ("POSE_KEYPOINT",),
                "padding_mode": (["auto-padding", "no-padding"],),
                "confidence_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "reset_confidence": ("BOOLEAN", {"default": False}),
                "force_normalized": ("BOOLEAN", {"default": False}),
                "pretty_json": ("BOOLEAN", {"default": False}),
                "save_output": ("BOOLEAN", {"default": False}),
            }
        }


    def execute(self, pose_keypoints: str, padding_mode: str, confidence_threshold: float,
                      reset_confidence: bool, force_normalized: bool, pretty_json: bool, save_output: bool):
        
        if isinstance(pose_keypoints, str):
            data = json.loads(pose_keypoints)
        else:
            data = copy.deepcopy(pose_keypoints)

        data = filter_keypoints_by_confidence(data, confidence_threshold, reset_confidence)
        data = remove_empty_keypoints(data)

        final_width = None
        final_height = None
        add_canvas_size_to_output = False

        if data and isinstance(data, list) and len(data) > 0:
            first_frame = data[0]
            keypoints_width = first_frame.get("canvas_width")
            keypoints_height = first_frame.get("canvas_height")

            if keypoints_width is not None and keypoints_height is not None:
                final_width = keypoints_width
                final_height = keypoints_height
        
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        has_keypoints = False

        for item in data:
            for person in item.get("people", []):
                for key in ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
                    if key in person:
                        pts = person[key]
                        if pts: has_keypoints = True
                        for i in range(0, len(pts), 3):
                            x, y = pts[i], pts[i+1]
                            min_x = min(min_x, x)
                            min_y = min(min_y, y)
                            max_x = max(max_x, x)
                            max_y = max(max_y, y)

        if not has_keypoints and (final_width is None or final_height is None):
            raise Exception("[KeypointConverter] ERROR: No keypoints found to determine canvas dimensions.")

        if padding_mode == "no-padding" and has_keypoints:
            final_width = max_x - min_x
            final_height = max_y - min_y
            for item in data:
                for person in item.get("people", []):
                    for key in ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
                        if key in person:
                            pts = person[key]
                            for i in range(0, len(pts), 3):
                                pts[i] -= min_x
                                pts[i+1] -= min_y
            add_canvas_size_to_output = True
        
        elif padding_mode == "auto-padding" and (final_width is None or final_height is None) and has_keypoints:
            final_width = max_x + min_x
            final_height = max_y + min_y
            add_canvas_size_to_output = True

        if add_canvas_size_to_output:
            if final_width is not None:
                final_width = int(((final_width + 7) // 8) * 8)
            if final_height is not None:
                final_height = int(((final_height + 7) // 8) * 8)
            
        if final_width is None or final_height is None:
            raise Exception("[KeypointConverter] ERROR: Canvas dimensions could not be determined "
                            "even after applying padding mode. This should not happen.")

        current_format = detect_coordinate_format(data)
        output_format = "normalized" if force_normalized else "absolute"
        converted_data = convert_pose(data, final_width, final_height, current_format, output_format, add_canvas_size_to_output)

        if pretty_json:
            converted_json_string = pretty_json_triplets_converter(converted_data)
        else:
            converted_json_string = json.dumps(converted_data, separators=(",", ":"))

        if save_output:
            output_dir = folder_paths.get_output_directory()
            os.makedirs(output_dir, exist_ok=True)
            i = 1
            while True:
                filename = f"keypoints_{output_format}_{i:04d}.json"
                filepath = os.path.join(output_dir, filename)
                if not os.path.exists(filepath):
                    break
                i += 1
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(converted_json_string)

        if pretty_json:
            return (converted_json_string,)
        else:
            return (converted_data,)

NODE_CLASS_MAPPINGS = {"KeypointConverter": KeypointConverter}

NODE_DISPLAY_NAME_MAPPINGS = {"KeypointConverter": "DWposeDeluxe Keypoint Converter"}