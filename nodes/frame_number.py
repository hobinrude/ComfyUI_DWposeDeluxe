
import torch
import numpy as np
import sys
import os

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False

def find_font_file(font_name):
    if os.path.isfile(font_name):
        return font_name
    
    font_dirs = []
    if sys.platform == "win32":
        font_dirs.append(os.path.join(os.environ['WINDIR'], 'Fonts'))
    elif sys.platform in ("linux", "linux2"):
        font_dirs.extend([
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            os.path.expanduser("~/.fonts"),
            os.path.expanduser("~/.local/share/fonts")
        ])
    elif sys.platform == "darwin":
        font_dirs.extend([
            "/System/Library/Fonts",
            "/Library/Fonts",
            os.path.expanduser("~/Library/Fonts")
        ])

    for dir_path in font_dirs:
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.lower() == font_name.lower():
                    return os.path.join(root, file)
    return None


class FrameNumberNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "frame_numbers": ("BOOLEAN", {"default": True}),
                "number_position": (["top-left", "top-right", "bottom-left", "bottom-right"],),
                "font_size": ("INT", {"default": 32, "min": 1, "max": 1024, "step": 1}),
                "font_name": ("STRING", {"default": "arial.ttf"}),
                "overlay_opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "DWPose Advanced/Utils"

    def _add_frame_number_overlay(self, image_np, frame_number, corner_position, font_name, font_size, opacity):
        if not HAS_PILLOW:
            raise ImportError("Pillow library is required for custom font rendering. Please install it with 'pip install Pillow'")

        # Convert main image to Pillow Image with alpha channel
        original_image = Image.fromarray(image_np).convert("RGBA")

        # Create a transparent overlay layer
        overlay = Image.new("RGBA", original_image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        font_path = find_font_file(font_name)
        if font_path is None:
            print(f"Warning: Font '{font_name}' not found. Falling back to default.")
            font = ImageFont.load_default()
        else:
            try:
                font = ImageFont.truetype(font_path, font_size)
            except IOError:
                print(f"Error: Could not load font '{font_name}' from path '{font_path}'. Falling back to default.")
                font = ImageFont.load_default()

        text = str(frame_number)
        
        # Use anchor='lt' to make positioning calculations simpler and more accurate
        bbox = draw.textbbox((0, 0), text, font=font, anchor='lt')
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        margin = int(min(original_image.height, original_image.width) / 20.0)

        # Calculate position based on the text's bounding box
        if corner_position == "top-left":
            pos = (margin, margin)
        elif corner_position == "top-right":
            pos = (original_image.width - text_w - margin, margin)
        elif corner_position == "bottom-left":
            pos = (margin, original_image.height - text_h - margin)
        else: # bottom-right
            pos = (original_image.width - text_w - margin, original_image.height - text_h - margin)

        shadow_offset = max(1, font_size // 16)
        shadow_pos = (pos[0] + shadow_offset, pos[1] + shadow_offset)
        
        # Draw shadow and text (fully opaque) on the separate overlay layer
        draw.text(shadow_pos, text, font=font, fill=(0, 0, 0, 255), anchor='lt')
        draw.text(pos, text, font=font, fill=(255, 255, 255, 255), anchor='lt')

        # Create an image with the text composited on it
        text_on_image = Image.alpha_composite(original_image, overlay)

        # Blend the original image with the text-composited image to apply final opacity
        final_image = Image.blend(original_image, text_on_image, alpha=opacity)

        # Convert back to a 3-channel RGB numpy array for ComfyUI
        return np.array(final_image.convert("RGB"))


    def execute(self, image, frame_numbers, number_position, font_size, font_name, overlay_opacity):
        if not frame_numbers:
            return (image,)
            
        if not HAS_PILLOW:
            raise ImportError("Pillow library is required for custom font rendering. Please install it with 'pip install Pillow'")

        batch_size = image.shape[0]
        output_frames = []

        for i in range(batch_size):
            img_np_hwc = (image[i].cpu().numpy() * 255).astype(np.uint8)
            
            numbered_frame = self._add_frame_number_overlay(
                img_np_hwc, 
                i + 1, 
                number_position,
                font_name,
                font_size,
                overlay_opacity
            )
            
            output_frames.append(numbered_frame)

        output_tensor = torch.from_numpy(np.array(output_frames).astype(np.float32) / 255.0)
        
        return (output_tensor,)

NODE_CLASS_MAPPINGS = {
    "FrameNumberNode": FrameNumberNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FrameNumberNode": "Frame Number Stamper"
}
