DWOPOSE_OPTIONS_TYPE = "DWOPOSE_OPTIONS"

class WeightOptions:
    RETURN_TYPES = (DWOPOSE_OPTIONS_TYPE,)
    RETURN_NAMES = ("options",)
    FUNCTION = "package_options"
    CATEGORY = "DWPose Advanced/Options"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "body_dot_size_modifier": ("INT", {"default": 0, "min": -3, "max": 3, "step": 1, "display": "slider"}),
                "body_line_thickness_modifier": ("INT", {"default": 0, "min": -3, "max": 3, "step": 1, "display": "slider"}),
                "hand_dot_size_modifier": ("INT", {"default": 0, "min": -3, "max": 3, "step": 1, "display": "slider"}),
                "hand_line_thickness_modifier": ("INT", {"default": 0, "min": -3, "max": 3, "step": 1, "display": "slider"}),
                "face_dot_size_modifier": ("INT", {"default": 0, "min": -3, "max": 3, "step": 1, "display": "slider"}),
            }
        }

    def package_options(self, **kwargs):
        return (kwargs,)

NODE_CLASS_MAPPINGS = {
    "WeightOptions": WeightOptions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WeightOptions": "DWposeDeluxe Weight Options",
}
