# ComfyUI_DWposeDeluxe/main.py

import os
import cv2
import sys
import time
import json
import torch
import numpy as np
import folder_paths

from comfy.utils import ProgressBar
from comfy.model_management import InterruptProcessingException

from . import HAS_TENSORRT
from .scripts import logger
from .scripts import progress
from .nodes.custom_options import DWOPOSE_CUSTOM_OPTIONS_TYPE


if HAS_TENSORRT:
    try:
        from .trt.trt_utilities import Engine
        from .trt.utilities import download_file, ColoredLogger
        import tensorrt
    except ImportError as e:
        logger.error(f"Failed to import TensorRT utilities:\n            {e}")
        pass

try:
    from .dwpose import DWposeDetector
    from .node_configs import DWposeNodeBase
    backend_available = True
    logger.info(f"Backend 'DWposeDetector' class loaded from .dwpose")
except ImportError as e:
    backend_available = False
    logger.error(f"Could not import backend from '.dwpose'\n            {e}")
    print("            Ensure './dwpose/__init__.py' defines 'DWposeDetector' and accepts parameters")
    print("            The node WILL FAIL during execution")


    class DWposeDetector:


        def __init__(self, provider_type: str, det_model_path: str, pose_model_path: str):
             self.is_dummy = True
             logger.info(f"Dummy DWposeDetector initialized (provider: {provider_type})")


        def __call__(self, oriImg, show_face=True, show_hands=True, show_feet=True, **render_options):
             logger.info(f"Dummy DWposeDetector called, returning input image")
             return oriImg


class DWposeDeluxeNode(DWposeNodeBase):
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "AUDIO", "FLOAT", "POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_image", "blend_image", "face_image", "source_image", "audio", "frame_rate", "keypoints",)
    FUNCTION = "execute"
    CATEGORY = "DWposeDeluxe"

    _detector_cache = {}


    @classmethod
    def INPUT_TYPES(cls):
        all_detector_models = set()
        all_estimator_models = set()

        try:
            onnx_detector_models, onnx_estimator_models = cls.get_model_list(provider_type="CPU")
            all_detector_models.update(onnx_detector_models)
            all_estimator_models.update(onnx_estimator_models)
        except Exception as e:
            logger.error(f"Could not get ONNX model lists for INPUT_TYPES\n             {e}")
            all_detector_models.add("auto-download")
            all_estimator_models.add("auto-download")

        if HAS_TENSORRT:
            for precision_type in ["fp16", "fp32"]:
                try:
                    trt_detector_models, trt_estimator_models = cls.get_model_list(provider_type="GPU", precision=precision_type)
                    all_detector_models.update(trt_detector_models)
                    all_estimator_models.update(trt_estimator_models)
                except Exception as e:
                    logger.error(f"Could not get TRT model lists for INPUT_TYPES (precision {precision_type})\n             {e}")
                    all_detector_models.add("auto-build")
                    all_estimator_models.add("auto-build")
        else:
            all_detector_models.add("TensorRT_Not_Installed")
            all_estimator_models.add("TensorRT_Not_Installed")
            all_detector_models.add("auto-build")
            all_estimator_models.add("auto-build")

        external_inputs = {
            "image": ("IMAGE",),
            "frame_count": ("INT", {"forceInput": True}),
            "audio": ("AUDIO", {"optional": True}),
            "video_info": ("VHS_VIDEOINFO", {"forceInput": True}),
            "custom_options": (DWOPOSE_CUSTOM_OPTIONS_TYPE,),
        }

        ui_inputs = {
            "poses_to_detect": ("INT", {"default": 1, "min": -1, "max": 100, "step": 1}),
            "show_body": ("BOOLEAN", {"default": True}),
            "show_feet": ("BOOLEAN", {"default": True}),
            "show_face": ("BOOLEAN", {"default": True}),
            "show_hands": ("BOOLEAN", {"default": True}),
            "crop_face": ("BOOLEAN", {"default": False}),
            "provider_type": (["CPU", "GPU"], {"default": "CPU"}),
            "precision": (["fp16", "fp32"], {"default": "fp32"}),
            "detector_model": (sorted(list(all_detector_models)), ),
            "estimator_model": (sorted(list(all_estimator_models)), ),
            "save_keypoints": ("BOOLEAN", {"default": False}),
        }

        inputs = {
            "required": {
                "image": external_inputs.pop("image"),
            },
            "optional": {
                **external_inputs,
                **ui_inputs,
            }
        }
        return inputs


    @classmethod
    def get_model_list(cls, provider_type, precision=None):
        models_dir = os.path.join(folder_paths.models_dir, "dwpose")

        onnx_detector_models = [f for f in os.listdir(models_dir) if f.startswith("yolox_") and f.endswith(".onnx")]
        onnx_estimator_models = [f for f in os.listdir(models_dir) if f.startswith("dw-ll_ucoco_") and f.endswith(".onnx")]

        detector_models = []
        estimator_models = []

        if provider_type == "CPU":
            detector_models = onnx_detector_models if onnx_detector_models else ["auto-download"]
            estimator_models = onnx_estimator_models if onnx_estimator_models else ["auto-download"]
        elif provider_type == "GPU":
            if not HAS_TENSORRT:
                detector_models = ["TensorRT_Not_Installed"]
                estimator_models = ["TensorRT_Not_Installed"]
            elif precision:
                if HAS_TENSORRT:
                    all_trt_files = [f for f in os.listdir(models_dir) if f.endswith(".trt")]

                    detector_prefix = f"yolox_l_{precision}"
                    matching_detector_models = [f for f in all_trt_files if f.startswith(detector_prefix) and f.endswith(".trt")]
                    detector_models = sorted(matching_detector_models) if matching_detector_models else ["auto-build"]

                    estimator_prefix = f"dw-ll_ucoco_384_{precision}"
                    matching_estimator_models = [f for f in all_trt_files if f.startswith(estimator_prefix) and f.endswith(".trt")]
                    estimator_models = sorted(matching_estimator_models) if matching_estimator_models else ["auto-build"]
                else:
                    detector_models = ["TensorRT_Not_Installed"]
                    estimator_models = ["TensorRT_Not_Installed"]
            else:
                detector_models = ["select_precision"]
                estimator_models = ["select_precision"]

        return (detector_models, estimator_models)


    @staticmethod
    def _get_model_list_api(provider_type, precision=None):
        detector_models, estimator_models = DWposeDeluxeNode.get_model_list(provider_type, precision)
        return {"detector_models": detector_models, "estimator_models": estimator_models}


    def _load_or_build_trt_engines(self, yolox_precision, dwpose_precision):
        import tensorrt
        models_dir = os.path.join(folder_paths.models_dir, "dwpose")

        os.makedirs(models_dir, exist_ok=True)

        yolox_onnx_model_path = os.path.join(models_dir, "yolox_l.onnx")
        dwpose_onnx_model_path = os.path.join(models_dir, "dw-ll_ucoco_384.onnx")
        
        yolox_tensorrt_model_path = os.path.join(models_dir, f"yolox_l_{yolox_precision}.trt")
        dwpose_tensorrt_model_path = os.path.join(models_dir, f"dw-ll_ucoco_384_{dwpose_precision}.trt")

        built_new_model = False

        if not os.path.exists(yolox_tensorrt_model_path):
            if not os.path.exists(yolox_onnx_model_path):
                logget.error(f"yolox_l onnx model not found at: {yolox_onnx_model_path}")
                raise InterruptProcessingException()
            
            logger.info(f"Building TensorRT engine for: {yolox_onnx_model_path}")
            logger.info(f"Saving TensorRT engine as   : {yolox_tensorrt_model_path}")
            engine = Engine(yolox_tensorrt_model_path)
            engine.build(
                onnx_path=yolox_onnx_model_path,
                fp16= True if yolox_precision == "fp16" else False
            )
            built_new_model = True

        if not os.path.exists(dwpose_tensorrt_model_path):
            if not os.path.exists(dwpose_onnx_model_path):
                logger.error(f"dwpose onnx model not found at: {dwpose_onnx_model_path}")
                raise InterruptProcessingException()

            logger.info(f"Building TensorRT engine for: {dwpose_onnx_model_path}")
            logger.info(f"Saving TensorRT engine as   : {dwpose_tensorrt_model_path}")
            engine = Engine(dwpose_tensorrt_model_path)
            engine.build(
                onnx_path=dwpose_onnx_model_path,
                fp16= True if dwpose_precision == "fp16" else False
            )
            built_new_model = True

        logger.info(f"Loading Yolox_L TensorRT engine: {yolox_tensorrt_model_path}")
        logger.info(f"Loading Dwpose TensorRT engine: {dwpose_tensorrt_model_path}")

        yolox_engine = Engine(yolox_tensorrt_model_path)
        dwpose_engine = Engine(dwpose_tensorrt_model_path)

        yolox_engine.load()
        dwpose_engine.load()

        return (yolox_engine, dwpose_engine, built_new_model,)


    def initialize_dwpose_detector(self, provider_type, precision, detector_model, estimator_model):
        built_new_model = False
        dwpose_detector = None

        if provider_type == "GPU":
            if precision is None:
                logger.error(f"Precision must be specified when provider_type is GPU")
                raise InterruptProcessingException()
            if not HAS_TENSORRT:
                logger.error(f"TensorRT is not available. Please install it to use TensorRT engine or use CPU mode")
                raise InterruptProcessingException()
            
            yolox_engine, dwpose_engine, built_new_model = self._load_or_build_trt_engines(precision, precision)
            
            dwpose_detector = DWposeDetector(
                provider_type=provider_type,
                det_model_path=None,
                pose_model_path=None,
                model_type="TensorRT",
                yolox_engine=yolox_engine,
                dwpose_engine=dwpose_engine
            )

        elif provider_type == "CPU":
            if not backend_available:
                logger.error(f"Backend logic could not be imported during startup")
                raise ImportError("Backend logic could not be imported during startup")
            try:
                det_path = folder_paths.get_full_path("dwpose", detector_model)
                est_path = folder_paths.get_full_path("dwpose", estimator_model)

                if not det_path or not os.path.exists(det_path):
                    logger.error(f"Detector model file not found: {detector_model}")
                    raise InterruptProcessingException()
                if not est_path or not os.path.exists(est_path):
                    logger.error(f"Estimator model file not found: {estimator_model}")
                    raise InterruptProcessingException()

            except Exception as e:
                logger.error(f"Could not find selected model files:\n            {e}")
                raise InterruptProcessingException()

            cache_key = (provider_type, det_path, est_path)

            if cache_key not in self._detector_cache:
                logger.info(f"Initializing backend DWposeDetector instance...")
                print(f" Detector       : {det_path}")
                print(f" Estimator      : {est_path}")
                try:

                    instance = DWposeDetector(
                        provider_type=provider_type,
                        det_model_path=det_path,
                        pose_model_path=est_path
                    )

                    if hasattr(instance, 'is_dummy'):
                        logger.error(f"Loaded backend instance is a dummy (import failed earlier)")
                        raise RuntimeError("Loaded backend instance is a dummy (import failed earlier)")

                    self._detector_cache[cache_key] = instance
                    logger.info(f"Backend instance created and cached")
                except Exception as e:
                    logger.error(f"Failed to instantiate DWposeDetector from .dwpose:\n            {e}")
                    raise RuntimeError(f"Could not create backend instance")
            dwpose_detector = self._detector_cache[cache_key]
        
        return dwpose_detector, built_new_model


    def process_frames(self, dwpose_detector, image, batch_size, height, width, 
                         show_body, show_face, show_hands, show_feet, crop_face, poses_to_detect,
                         actual_options,
                         provider_type, precision):

        logger.info(f"Limiting pose detections to top {poses_to_detect} people by bounding box area")
        logger.info(f"Processing {batch_size} frames of size {width}x{height}" + (f" using TensorRT ({precision})" if provider_type == "GPU" else ""))
        
        pbar = progress(batch_size, label="Frame")

        results_list = []
        all_keypoints_data = []

        face_data_needed = show_face or crop_face

        for i in range(batch_size):
            img_np_hwc = (image[i].cpu().numpy() * 255).astype(np.uint8)
            if img_np_hwc.shape[2] == 4: img_np_hwc = img_np_hwc[:, :, :3]
            elif img_np_hwc.shape[2] == 1: img_np_hwc = np.repeat(img_np_hwc, 3, axis=2)

            try:
                processed_np, keypoints_data = dwpose_detector(
                    oriImg=img_np_hwc,
                    show_body=show_body,
                    show_face=show_face,
                    show_hands=show_hands,
                    show_feet=show_feet,
                    _calculate_face=face_data_needed,
                    poses_to_detect=poses_to_detect,
                    **actual_options
                )

                keypoints_data['canvas_width'] = width
                keypoints_data['canvas_height'] = height
                all_keypoints_data.append(keypoints_data)
            except Exception as e:
                logger.error(f"Backend DWposeDetector call failed on frame {i+1}: {e}")
                import traceback
                traceback.print_exc()
                processed_np = np.zeros_like(img_np_hwc)

            results_list.append(processed_np)

            pbar.step()

        pbar.finish()
            
        if provider_type == "GPU":
            dwpose_detector.reset_engines()

        return results_list, all_keypoints_data


    def get_face_cropping_metadata(self, all_keypoints_data, face_padding):
        if not all_keypoints_data:
            return {}, 0, 0

        def round_to_nearest_multiple(n, m=8):
            return int(round(n / m) * m) if n > 0 else 0

        master_size = 0
        all_square_dims = []
        faces_by_frame = {i: [] for i in range(len(all_keypoints_data))}
        max_faces_per_frame = 0

        for i, keypoints_data in enumerate(all_keypoints_data):
            current_frame_faces_temp = []
            if 'people' in keypoints_data:
                for person in keypoints_data['people']:
                    if 'face_box' in person:
                        x_min, y_min, x_max, y_max = person['face_box']
                        w, h = x_max - x_min, y_max - y_min
                        
                        padded_w = w + (2 * face_padding)
                        padded_h = h + (2 * face_padding)
                        square_dim = max(padded_w, padded_h)
                        
                        face_detail = {'face_box': person['face_box'], 'square_dim': square_dim}
                        current_frame_faces_temp.append(face_detail)
                        all_square_dims.append(square_dim)
            
            current_frame_faces_temp.sort(key=lambda x: x['face_box'][0])
            faces_by_frame[i] = current_frame_faces_temp
            if len(current_frame_faces_temp) > max_faces_per_frame:
                max_faces_per_frame = len(current_frame_faces_temp)

        if all_square_dims:
            biggest_square_dim = max(all_square_dims)
            master_size = round_to_nearest_multiple(biggest_square_dim)
        
        return faces_by_frame, master_size, max_faces_per_frame


    def generate_face_atlas_cpu(self, image_tensor, all_keypoints_data, face_padding, height, width):
        faces_by_frame, master_size, max_faces_per_frame = self.get_face_cropping_metadata(all_keypoints_data, face_padding)
        if master_size == 0:
            return torch.zeros((image_tensor.shape[0], 1, 1, 3), dtype=torch.float32)

        batch_size = image_tensor.shape[0]
        logger.info(f"Cropping faces from {batch_size} frames with padding={face_padding} using CPU.")

        pbar = progress(batch_size, label="Frame")

        final_atlas_width = master_size * max_faces_per_frame
        final_atlas_np = np.zeros((batch_size, master_size, final_atlas_width, 3), dtype=np.uint8)

        for i in range(image_tensor.shape[0]):
            original_img_np = (image_tensor[i].cpu().numpy() * 255).astype(np.uint8)
            if original_img_np.shape[2] == 4: original_img_np = original_img_np[:, :, :3]
            elif original_img_np.shape[2] == 1: original_img_np = np.repeat(original_img_np, 3, axis=2)

            current_frame_faces = faces_by_frame.get(i, [])
            for face_idx, detail in enumerate(current_frame_faces):
                x_min, y_min, x_max, y_max = detail['face_box']
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                square_dim = detail['square_dim']

                src_x1 = int(cx - square_dim / 2)
                src_y1 = int(cy - square_dim / 2)
                src_x2 = int(cx + square_dim / 2)
                src_y2 = int(cy + square_dim / 2)

                src_x1_clip = max(0, src_x1)
                src_y1_clip = max(0, src_y1)
                src_x2_clip = min(width, src_x2)
                src_y2_clip = min(height, src_y2)
                
                source_crop = original_img_np[src_y1_clip:src_y2_clip, src_x1_clip:src_x2_clip]

                if source_crop.size > 0:
                    pad_top = src_y1_clip - src_y1
                    pad_bottom = src_y2 - src_y2_clip
                    pad_left = src_x1_clip - src_x1
                    pad_right = src_x2 - src_x2_clip
                    
                    padded_crop = cv2.copyMakeBorder(source_crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    scaled_crop = cv2.resize(padded_crop, (master_size, master_size), interpolation=cv2.INTER_AREA)

                    start_x = face_idx * master_size
                    end_x = start_x + master_size
                    final_atlas_np[i, :, start_x:end_x, :] = scaled_crop

            pbar.step()
            
        pbar.finish()
        return torch.from_numpy(final_atlas_np.astype(np.float32) / 255.0)


    def generate_face_atlas_gpu(self, image_tensor, all_keypoints_data, face_padding, height, width):
        faces_by_frame, master_size, max_faces_per_frame = self.get_face_cropping_metadata(all_keypoints_data, face_padding)
        if master_size == 0:
            return torch.zeros((image_tensor.shape[0], 1, 1, 3), device=image_tensor.device, dtype=torch.float32)

        batch_size = image_tensor.shape[0]
        logger.info(f"Cropping faces from {batch_size} frames with padding={face_padding} using GPU.")

        pbar = progress(batch_size, label="Frame")

        final_atlas_width = master_size * max_faces_per_frame
        final_atlas_tensor = torch.zeros((batch_size, master_size, final_atlas_width, 3), device=image_tensor.device, dtype=image_tensor.dtype)
        
        for i in range(image_tensor.shape[0]):
            current_frame_faces = faces_by_frame.get(i, [])
            for face_idx, detail in enumerate(current_frame_faces):
                x_min, y_min, x_max, y_max = detail['face_box']
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                square_dim = int(detail['square_dim'])

                src_x1 = int(cx - square_dim / 2)
                src_y1 = int(cy - square_dim / 2)

                src_x1_clip = max(0, src_x1)
                src_y1_clip = max(0, src_y1)
                src_x2_clip = min(width, src_x1 + square_dim)
                src_y2_clip = min(height, src_y1 + square_dim)
                
                source_crop_tensor = image_tensor[i, src_y1_clip:src_y2_clip, src_x1_clip:src_x2_clip, :]

                if source_crop_tensor.numel() > 0:
                    padded_crop = torch.zeros((square_dim, square_dim, 3), device=image_tensor.device, dtype=image_tensor.dtype)
                    pad_top = src_y1_clip - src_y1
                    pad_left = src_x1_clip - src_x1
                    padded_crop[pad_top:pad_top+source_crop_tensor.shape[0], pad_left:pad_left+source_crop_tensor.shape[1], :] = source_crop_tensor

                    resize_input = padded_crop.permute(2, 0, 1).unsqueeze(0)
                    scaled_crop_tensor = torch.nn.functional.interpolate(resize_input, size=(master_size, master_size), mode='area')
                    final_crop = scaled_crop_tensor.squeeze(0).permute(1, 2, 0)
                    
                    start_x = face_idx * master_size
                    end_x = start_x + master_size
                    final_atlas_tensor[i, :, start_x:end_x, :] = final_crop

            pbar.step()

        pbar.finish()
        return final_atlas_tensor


    def validate_and_prepare_frame(self, frame, height, width):
        if frame is None or not isinstance(frame, np.ndarray):
            return np.zeros((height, width, 3), dtype=np.uint8)
        if frame.ndim == 2: frame = np.stack((frame,) * 3, axis=-1)
        if frame.shape[2] == 4: frame = frame[:, :, :3]
        if frame.shape[2] == 1: frame = np.repeat(frame, 3, axis=2)
        if frame.ndim != 3 or frame.shape[2] != 3:
            return np.zeros((height, width, 3), dtype=np.uint8)
        return frame.astype(np.uint8)


    def generate_pose_and_blend_tensors_gpu(self, image_tensor, results_list):
        batch_size, height, width, _ = image_tensor.shape

        logger.info(f"Generating pose and blend image_tensors using GPU")

        pbar = progress(batch_size, label="Frame")

        device = image_tensor.device
        dtype = image_tensor.dtype

        pose_output_tensor = torch.zeros_like(image_tensor)
        blended_output_tensor = torch.zeros_like(image_tensor)
        
        for i in range(batch_size):
            pose_np = self.validate_and_prepare_frame(results_list[i], height, width)
            pose_tensor = torch.from_numpy(pose_np.astype(np.float32) / 255.0).to(device, dtype=dtype)
            
            source_tensor_slice = image_tensor[i]

            if pose_tensor.shape[0] != height or pose_tensor.shape[1] != width:
                pose_tensor_nchw = pose_tensor.unsqueeze(0).permute(0, 3, 1, 2)
                resized_pose_nchw = torch.nn.functional.interpolate(pose_tensor_nchw, size=(height, width), mode='area')
                resized_pose_tensor = resized_pose_nchw.permute(0, 2, 3, 1).squeeze(0)
            else:
                resized_pose_tensor = pose_tensor

            blended_slice = source_tensor_slice * 0.5 + resized_pose_tensor * 0.5
            pose_output_tensor[i] = resized_pose_tensor
            blended_output_tensor[i] = blended_slice

            pbar.step()

        pbar.finish()
        return pose_output_tensor, blended_output_tensor, image_tensor


    def generate_pose_and_blend_tensors_cpu(self, image_tensor, results_list):
        batch_size, height, width, _ = image_tensor.shape

        logger.info(f"Generating pose and blend image_tensors using CPU")

        pbar = progress(batch_size, label="Frame")

        pose_output_frames = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
        blended_output_frames = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
        source_output_frames = np.zeros((batch_size, height, width, 3), dtype=np.uint8)

        for i in range(batch_size):
            source_np = (image_tensor[i].cpu().numpy() * 255).astype(np.uint8)
            pose_np = self.validate_and_prepare_frame(results_list[i], height, width)

            if pose_np.shape[0] != height or pose_np.shape[1] != width:
                resized_pose_np = cv2.resize(pose_np, (width, height), interpolation=cv2.INTER_AREA)
            else:
                resized_pose_np = pose_np

            blended_np = cv2.addWeighted(source_np, 0.5, resized_pose_np, 0.5, 0)
            
            pose_output_frames[i] = resized_pose_np
            blended_output_frames[i] = blended_np
            source_output_frames[i] = source_np

            pbar.step()

        pbar.finish()

        pose_output_tensor = torch.from_numpy(pose_output_frames.astype(np.float32) / 255.0)
        blended_output_tensor = torch.from_numpy(blended_output_frames.astype(np.float32) / 255.0)
        source_output_tensor = torch.from_numpy(source_output_frames.astype(np.float32) / 255.0)

        return pose_output_tensor, blended_output_tensor, source_output_tensor


    def generate_outputs(self, image, results_list, all_keypoints_data, show_face, crop_face, height, width, save_keypoints, audio, fps_to_output, built_new_model, face_padding, provider_type):
        
        if crop_face:
            if provider_type == "GPU":
                face_atlas_tensor = self.generate_face_atlas_gpu(image, all_keypoints_data, face_padding, height, width)
            else: # CPU
                face_atlas_tensor = self.generate_face_atlas_cpu(image, all_keypoints_data, face_padding, height, width)
        else:
            face_atlas_tensor = torch.zeros((image.shape[0], 1, 1, 3), dtype=torch.float32)

        if provider_type == "GPU":
            pose_output_tensor, blended_output_tensor, source_output_tensor = self.generate_pose_and_blend_tensors_gpu(image, results_list)
        else: # CPU
            pose_output_tensor, blended_output_tensor, source_output_tensor = self.generate_pose_and_blend_tensors_cpu(image, results_list)

        keypoints_json_string = json.dumps(all_keypoints_data)

        if save_keypoints:
            output_dir = folder_paths.get_output_directory()
            os.makedirs(output_dir, exist_ok=True)
            i = 1
            while True:
                filename = f"keypoints_absolute_{i:04d}.json"
                filepath = os.path.join(output_dir, filename)
                if not os.path.exists(filepath):
                    break
                i += 1
            with open(filepath, "w") as f:
                f.write(keypoints_json_string)
        sys.stdout.write('\n')
        ui_output = {}
        if built_new_model:
            ui_output = {"ui": {"model_refresh_needed": True}}
        return {"ui": ui_output, "result": (pose_output_tensor, blended_output_tensor, face_atlas_tensor, source_output_tensor, audio, fps_to_output, all_keypoints_data,)}


    def execute(self, image: torch.Tensor, provider_type: str,
                      frame_count: int = 0, audio: torch.Tensor = None, video_info: str = "",
                      precision: str = "fp32",
                      detector_model: str = "None", estimator_model: str = "None",
                      show_body: bool = True, show_face: bool = True, show_hands: bool = True, show_feet: bool = True, crop_face: bool = False, save_keypoints: bool = False,
                      poses_to_detect: int = 1,
                      custom_options: dict = None, **kwargs):

        fps_to_output = float('nan')
        if not video_info:
            logger.warning(f"No video_info input connected so frame_rate output will produce ERROR when used")
        else:
            video_info_data = None
            if isinstance(video_info, dict):
                video_info_data = video_info
            else:
                try:
                    video_info_data = json.loads(video_info)
                except (json.JSONDecodeError, TypeError):
                    pass
            if video_info_data:
                loaded_fps = video_info_data.get("loaded_fps")
                if loaded_fps is not None:
                    fps_to_output = loaded_fps
        logger.info(f"Using {fps_to_output} FPS for output")

        actual_options = {}
        default_options = {
            "body_dot_size_modifier": 0, "body_line_thickness_modifier": 0,
            "hand_dot_size_modifier": 0, "hand_line_thickness_modifier": 0,
            "face_dot_size_modifier": 0,
            "pose_threshold": 0.25,
            "body_threshold": 0.30,
            "face_threshold": 0.10,
            "hand_threshold": 0.10,
            "face_padding": 0,
            "neck_validity": 0.30,
            "nms_threshold": 0.45,
            "score_threshold": 0.10,
        }
        actual_options.update(default_options)

        if custom_options is not None:
            actual_options.update(custom_options)
            logger.info(f"Received options from node:")
            print(f"Body dot size modifier       : {actual_options['body_dot_size_modifier']}")
            print(f"Body bone thickness modifier : {actual_options['body_line_thickness_modifier']}")
            print(f"Hand dot size modifier       : {actual_options['hand_dot_size_modifier']}")
            print(f"Hand bone thickness modifier : {actual_options['hand_line_thickness_modifier']}")
            print(f"Face dot size modifier       : {actual_options['face_dot_size_modifier']}")
            print(f"Pose threshold               : {actual_options['pose_threshold']}")
            print(f"Body threshold               : {actual_options['body_threshold']}")
            print(f"Face threshold               : {actual_options['face_threshold']}")
            print(f"Hand threshold               : {actual_options['hand_threshold']}")
            print(f"Face padding                 : {actual_options['face_padding']}")
        else:
            logger.warning(f"CustomOptions node not connected, using default modifiers and thresholds")

        pose_threshold = actual_options["pose_threshold"]
        body_threshold = actual_options["body_threshold"]
        face_threshold = actual_options["face_threshold"]
        hand_threshold = actual_options["hand_threshold"]
        face_padding = actual_options["face_padding"]

        dwpose_detector, built_new_model = self.initialize_dwpose_detector(provider_type, precision, detector_model, estimator_model)
        if provider_type == "GPU":
            dwpose_detector.activate_engines()

        batch_size = image.shape[0]
        height, width = image.shape[1], image.shape[2]
        results_list, all_keypoints_data = self.process_frames(
            dwpose_detector=dwpose_detector,
            image=image,
            batch_size=batch_size,
            height=height,
            width=width,
            show_body=show_body,
            show_face=show_face,
            show_hands=show_hands,
            show_feet=show_feet,
            crop_face=crop_face,
            poses_to_detect=poses_to_detect,
            actual_options=actual_options,
            provider_type=provider_type,
            precision=precision
        )

        outputs = self.generate_outputs(
            image=image,
            results_list=results_list,
            all_keypoints_data=all_keypoints_data,
            show_face=show_face,
            crop_face=crop_face,
            height=height,
            width=width,
            save_keypoints=save_keypoints,
            audio=audio,
            fps_to_output=fps_to_output,
            built_new_model=built_new_model,
            face_padding=actual_options["face_padding"],
            provider_type=provider_type
        )
        return outputs


NODE_CLASS_MAPPINGS = {
}

NODE_DISPLAY_NAME_MAPPINGS = {
}

if HAS_TENSORRT:
    NODE_CLASS_MAPPINGS["DWposeDeluxeNode"] = DWposeDeluxeNode
    NODE_DISPLAY_NAME_MAPPINGS["DWposeDeluxeNode"] = "DWposeDeluxe Estimator"
else:
    NODE_CLASS_MAPPINGS["DWposeDeluxeNode"] = DWposeDeluxeNode
    NODE_DISPLAY_NAME_MAPPINGS["DWposeDeluxeNode"] = "DWposeDeluxe Estimator (No TRT)"