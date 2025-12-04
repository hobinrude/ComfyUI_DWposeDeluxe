# ComfyUI_DWposeDeluxe/nodes/custom_options.py

DWOPOSE_CUSTOM_OPTIONS_TYPE = "DWOPOSE_CUSTOM_OPTIONS"


class CustomOptions:
    RETURN_TYPES = (DWOPOSE_CUSTOM_OPTIONS_TYPE,)
    RETURN_NAMES = ("custom_options",)
    FUNCTION = "package_options"
    CATEGORY = "DWposeDeluxe Options"


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "body_dot_size_modifier": ("INT", {"default": 0, "min": -3, "max": 6, "step": 1, "display": "slider"}),
                "body_line_thickness_modifier": ("INT", {"default": 0, "min": -3, "max": 6, "step": 1, "display": "slider"}),
                "hand_dot_size_modifier": ("INT", {"default": 0, "min": -3, "max": 6, "step": 1, "display": "slider"}),
                "hand_line_thickness_modifier": ("INT", {"default": 0, "min": -3, "max": 6, "step": 1, "display": "slider"}),
                "face_dot_size_modifier": ("INT", {"default": 0, "min": -3, "max": 6, "step": 1, "display": "slider"}),
                "pose_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "body_threshold": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01}),
                "face_threshold": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "hand_threshold": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "face_padding": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }


    def package_options(self, **kwargs):
        return (kwargs,)

NODE_CLASS_MAPPINGS = {"CustomOptions": CustomOptions}

NODE_DISPLAY_NAME_MAPPINGS = {"CustomOptions": "DWposeDeluxe Custom Options"}