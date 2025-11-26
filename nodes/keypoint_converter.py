import json
import os
import folder_paths
import copy

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


def convert_pose(data, width, height, fmt, add_canvas_size_to_output=True):
    for item in data:
        for person in item.get("people", []):
            for key in [
                "pose_keypoints_2d",
                "face_keypoints_2d",
                "hand_left_keypoints_2d",
                "hand_right_keypoints_2d",
            ]:
                if key in person: # Only process keys that exist in the 'person' dictionary
                    pts = person[key] # Get the existing list, which is guaranteed not to be empty
                    for i in range(0, len(pts), 3):
                        if fmt == "absolute":
                            pts[i] = pts[i] / width
                            pts[i + 1] = pts[i + 1] / height
                            pts[i + 2] = 1.0
                        else:
                            pts[i] = pts[i] * width
                            pts[i + 1] = pts[i + 1] * height

                    person[key] = pts
        if add_canvas_size_to_output:
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
            people_block = '\n        "people": [\n' + ",\n".join(people_list_str) + '\n        ]'
            frame_parts.append(people_block)
        other_keys = [k for k in frame.keys() if k != "people"]
        for key in other_keys:
            frame_parts.append(f'        \"{key}\": {json.dumps(frame[key])}')
        output_frames.append("    {\n" + ",\n".join(output_frames) + "\n    }")
    return "[\n" + ",\n".join(output_frames) + "\n]"

def remove_empty_keypoints(data):
    cleaned_data = []
    for frame in data:
        new_frame = {} # Start with an empty dict
        for key, value in frame.items(): # Iterate in original order
            if key == "people":
                cleaned_people = []
                for person in value: # `value` is the list of people
                    cleaned_person = {k: v for k, v in person.items() if not (isinstance(v, list) and not v)}
                    if cleaned_person:
                        cleaned_people.append(cleaned_person)
                if cleaned_people: # Only add the "people" key if it's not empty after cleaning
                    new_frame["people"] = cleaned_people
            else:
                # For all other keys (like canvas_width), just copy them over
                new_frame[key] = value
        cleaned_data.append(new_frame)
    return cleaned_data


class KeypointConverter:
    RETURN_TYPES = ("POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_keypoints",)
    FUNCTION = "execute"
    CATEGORY = "DWPose Advanced/Utils"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "custom_canvas_size": ("BOOLEAN", {"default": False}),
                "pose_keypoints": ("POSE_KEYPOINT",),
                "canvas_width": ("INT", {"default": 768, "min": 1, "max": 4096, "step": 1}),
                "canvas_height": ("INT", {"default": 768, "min": 1, "max": 4096, "step": 1}),
                "human_friendly_json": ("BOOLEAN", {"default": True}),
                "save_output": ("BOOLEAN", {"default": False}),
            }
        }

    def execute(self, custom_canvas_size: bool, pose_keypoints: str, canvas_width: int, canvas_height: int,
                      human_friendly_json: bool, save_output: bool):
        
        if isinstance(pose_keypoints, str):
            data = json.loads(pose_keypoints)
        else:
            data = copy.deepcopy(pose_keypoints)

        # Remove any key-value pairs where the value is an empty list.
        data = remove_empty_keypoints(data)

        final_width = None
        final_height = None
        add_canvas_size_to_output = custom_canvas_size

        if custom_canvas_size:
            final_width = canvas_width
            final_height = canvas_height
        else:
            if data and isinstance(data, list) and len(data) > 0:
                first_frame = data[0]
                keypoints_width = first_frame.get("canvas_width")
                keypoints_height = first_frame.get("canvas_height")

                if keypoints_width is not None and keypoints_height is not None:
                    final_width = keypoints_width
                    final_height = keypoints_height

        if final_width is None or final_height is None:
            raise Exception("[KeypointConverter] ERROR: Canvas dimensions could not be determined. "
                          "Enable 'custom_canvas_size' or ensure the input keypoints contain 'canvas_width' and 'canvas_height'.")

        current_format = detect_format(data)
        if current_format == "unknown":
            converted_data = data
            output_format = "unknown"
        else:
            converted_data = convert_pose(data, final_width, final_height, current_format, add_canvas_size_to_output)
            output_format = "relative" if current_format == "absolute" else "absolute"

        if human_friendly_json:
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

        if human_friendly_json:
            return (converted_json_string,)
        else:
            return (converted_data,)

NODE_CLASS_MAPPINGS = {
    "KeypointConverter": KeypointConverter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KeypointConverter": "DWposeDeluxe Keypoint Converter",
}
