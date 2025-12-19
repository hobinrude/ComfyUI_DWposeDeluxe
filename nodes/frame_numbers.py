# ComfyUI_DWposeDeluxe/nodes/frame_number.py

from ..node_configs import DWposeNodeBase
import torch
import numpy as np
import sys
import os
from PIL import Image, ImageDraw, ImageFont


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


class FrameNumberNode(DWposeNodeBase):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "print_frame_numbers": ("BOOLEAN", {"default": True}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "step": 1}),
                "print_every_nth": ("INT", {"default": 1, "min": 1, "step": 1}),
                "number_position": (["top-left", "top-right", "bottom-left", "bottom-right"],),
                "font_size": ("INT", {"default": 64, "min": 1, "max": 1024, "step": 1}),
                "font_name": ("STRING", {"default": "DejaVuSansMono-Bold.ttf"}),
                "overlay_opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "DWposeDeluxe"


    def _add_frame_number_overlay(self, image_np, frame_number, corner_position, font_name, font_size, opacity):

        original_image = Image.fromarray(image_np).convert("RGBA")

        overlay = Image.new("RGBA", original_image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        font_path = find_font_file(font_name)
        if font_path is None:
            logger.warning(f"[FrameNumbering] Font '{font_name}' not found in '{font_path}'. Falling back to default.")
            font = ImageFont.load_default()
        else:
            try:
                font = ImageFont.truetype(font_path, font_size)
            except IOError:
                logger.error(f"[FrameNumbering] Could not load font '{font_name}' from path '{font_path}'. Falling back to default.")
                font = ImageFont.load_default()

        text = str(frame_number)
        
        bbox = draw.textbbox((0, 0), text, font=font, anchor='lt')
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        margin = int(min(original_image.height, original_image.width) / 20.0)

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
        
        draw.text(shadow_pos, text, font=font, fill=(0, 0, 0, 255), anchor='lt')
        draw.text(pos, text, font=font, fill=(255, 255, 255, 255), anchor='lt')

        text_on_image = Image.alpha_composite(original_image, overlay)

        final_image = Image.blend(original_image, text_on_image, alpha=opacity)

        return np.array(final_image.convert("RGB"))


    def execute(self, image, skip_first_frames, print_every_nth, print_frame_numbers, number_position, font_size, font_name, overlay_opacity):
        if not print_frame_numbers:
            return (image,)

        batch_size = image.shape[0]
        output_frames = []

        number_offset = skip_first_frames
        number_increment = print_every_nth

        for i in range(batch_size):
            img_np_hwc = (image[i].cpu().numpy() * 255).astype(np.uint8)
            
            number_to_print = number_offset + (i * number_increment)
            numbered_frame = self._add_frame_number_overlay(
                img_np_hwc, 
                number_to_print, 
                number_position,
                font_name,
                font_size,
                overlay_opacity
            )
            output_frames.append(numbered_frame)

        output_tensor = torch.from_numpy(np.array(output_frames).astype(np.float32) / 255.0)
        
        return (output_tensor,)

NODE_CLASS_MAPPINGS = {"FrameNumberNode": FrameNumberNode}
NODE_DISPLAY_NAME_MAPPINGS = {"FrameNumberNode": "DWposeDeluxe Frame Numbering"}