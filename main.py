# ComfyUI_DWposeDeluxe/nodes.py

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
from .nodes.weight_options import DWOPOSE_OPTIONS_TYPE


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


class DWposeDeluxeNode:
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "AUDIO", "FLOAT", "POSE_KEYPOINT",)
    RETURN_NAMES = ("pose_image", "blend_image", "source_image", "audio", "frame_rate", "keypoints",)
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
            "weight_options": (DWOPOSE_OPTIONS_TYPE,),
        }

        ui_inputs = {
            "poses_to_detect": ("INT", {"default": 1, "min": -1, "max": 100, "step": 1}),
            "show_body": ("BOOLEAN", {"default": True}),
            "show_feet": ("BOOLEAN", {"default": True}),
            "show_face": ("BOOLEAN", {"default": True}),
            "show_hands": ("BOOLEAN", {"default": True}),
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


    def execute(self, image: torch.Tensor, provider_type: str,
                      frame_count: int = 0, audio: torch.Tensor = None, video_info: str = "",
                      precision: str = "fp32",
                      detector_model: str = "None", estimator_model: str = "None",
                      show_body: bool = True, show_face: bool = True, show_hands: bool = True, show_feet: bool = True, save_keypoints: bool = False,
                      poses_to_detect: int = 1,
                      weight_options: dict = None, **kwargs):

        fps_to_output = float('nan')
        if not video_info:
            logger.warning(f"No video_info input connected so frame_rate output will produce ERROR in downstream connected node when used")
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
        logger.info(f"Using FPS for output: {fps_to_output}")

        actual_options = {}
        default_options = {
            "body_dot_size_modifier": 0, "body_line_thickness_modifier": 0,
            "hand_dot_size_modifier": 0, "hand_line_thickness_modifier": 0,
            "face_dot_size_modifier": 0,
        }
        actual_options.update(default_options)

        if weight_options is not None:
            actual_options.update(weight_options)
            logger.info(f"Received options from node:")
            print(f"Body dot size modifier       : {actual_options['body_dot_size_modifier']}")
            print(f"Body bone thickness modifier : {actual_options['body_line_thickness_modifier']}")
            print(f"Hand dot size modifier       : {actual_options['hand_dot_size_modifier']}")
            print(f"Hand bone thickness modifier : {actual_options['hand_line_thickness_modifier']}")
            print(f"Face dot size modifier       : {actual_options['face_dot_size_modifier']}")
        else:
            logger.warning(f"WeightOptions node not connected, using default modifiers")

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

            dwpose_detector.activate_engines()

            logger.info(f"Limiting pose detections to top {poses_to_detect} people by bounding box area")

            batch_size = image.shape[0]
            height, width = image.shape[1], image.shape[2]
            logger.info(f"Processing {batch_size} frames of size {width}x{height} using TensorRT ({precision})")
            pbar = ProgressBar(batch_size)
            start_time = time.time()
            results_list = []
            all_keypoints_data = []

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
                        poses_to_detect=poses_to_detect,
                        **actual_options
                    )

                    if 'people' in keypoints_data:
                        for person in keypoints_data['people']:
                            if not show_face:
                                person.pop('face_keypoints_2d', None)
                            if not show_hands:
                                person.pop('hand_left_keypoints_2d', None)
                                person.pop('hand_right_keypoints_2d', None)

                    keypoints_data['canvas_width'] = width
                    keypoints_data['canvas_height'] = height
                    all_keypoints_data.append(keypoints_data)
                except Exception as e:
                    logger.error(f"TRT Backend DWposeDetector call failed on frame {i+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    processed_np = np.zeros_like(img_np_hwc)

                results_list.append(processed_np)
                pbar.update(1)
                progress = (i + 1) / batch_size
                bar = '█' * int(50 * progress) + '░' * (50 - int(50 * progress))
                percentage = int(progress * 100)

                # --- Calculate in-flight FPS ---
                elapsed_time = time.time() - start_time
                if elapsed_time > 0.001:
                    current_fps = (i + 1) / elapsed_time
                    fps_text = f" @ {current_fps:.2f} FPS"
                else:
                    fps_text = ""

                sys.stdout.write(f'\r{percentage}% {bar} 100% | Processing frame {i + 1}/{batch_size}{fps_text}')
                sys.stdout.flush()
            dwpose_detector.reset_engines()
            
            if not results_list:
                return (torch.empty((0, height, width, 3), dtype=torch.float32),)

            validated_frames = []
            for frame in results_list:
                if frame is None or not isinstance(frame, np.ndarray):
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                elif frame.ndim == 2: frame = np.stack((frame,) * 3, axis=-1)
                elif frame.shape[2] == 4: frame = frame[:, :, :3]
                elif frame.shape[2] == 1: frame = np.repeat(frame, 3, axis=2)
                if frame.ndim != 3 or frame.shape[2] != 3:
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                validated_frames.append(frame.astype(np.uint8))

            pose_output_frames = []
            blended_output_frames = []
            source_output_frames = []

            for i in range(batch_size):

                original_img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
                if original_img_np.shape[2] == 4: original_img_np = original_img_np[:, :, :3]
                elif original_img_np.shape[2] == 1: original_img_np = np.repeat(original_img_np, 3, axis=2)

                current_pose_img_np = validated_frames[i]
                pose_output_frames.append(current_pose_img_np)

                resized_pose_for_blend = current_pose_img_np
                if original_img_np.shape != resized_pose_for_blend.shape:
                    resized_pose_for_blend = cv2.resize(resized_pose_for_blend, (original_img_np.shape[1], original_img_np.shape[0]))
                blended_img_np = cv2.addWeighted(original_img_np, 0.5, resized_pose_for_blend, 0.5, 0)
                blended_output_frames.append(blended_img_np)

                current_source_img_np = original_img_np.copy()
                source_output_frames.append(current_source_img_np)

            pose_output_tensor = torch.from_numpy(np.array(pose_output_frames).astype(np.float32) / 255.0)
            blended_output_tensor = torch.from_numpy(np.array(blended_output_frames).astype(np.float32) / 255.0)
            source_output_tensor = torch.from_numpy(np.array(source_output_frames).astype(np.float32) / 255.0)

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
            logger.info(f"Built new model: {built_new_model}")
            return {"ui": ui_output, "result": (pose_output_tensor, blended_output_tensor, source_output_tensor, audio, fps_to_output, all_keypoints_data,)}
            
        if not backend_available:
             logger.error(f"Backend logic could not be imported during startup")
             raise ImportError("[DWposeNode][ERROR] Backend logic could not be imported during startup")
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
            print(f"    Provider: {provider_type}")
            print(f"    Detector: {det_path}")
            print(f"   Estimator: {est_path}")
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
        logger.info(f"Limiting pose detections to top {poses_to_detect} people by bounding box area")

        batch_size = image.shape[0]
        height, width = image.shape[1], image.shape[2]
        logger.info(f"Processing {batch_size} frames of size {width}x{height}")
        
        pbar = ProgressBar(batch_size)
        start_time = time.time()
        results_list = []

        all_keypoints_data = []
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
                     poses_to_detect=poses_to_detect,
                     **actual_options
                 )

                 if 'people' in keypoints_data:
                     for person in keypoints_data['people']:
                         if not show_face:
                             person.pop('face_keypoints_2d', None)
                         if not show_hands:
                             person.pop('hand_left_keypoints_2d', None)
                             person.pop('hand_right_keypoints_2d', None)

                 keypoints_data['canvas_width'] = width
                 keypoints_data['canvas_height'] = height
                 all_keypoints_data.append(keypoints_data)
            except Exception as e:
                 logger.error(f"Backend DWposeDetector call failed on frame {i+1}: {e}")
                 import traceback
                 traceback.print_exc()
                 processed_np = np.zeros_like(img_np_hwc)

            results_list.append(processed_np)
            pbar.update(1)

            progress = (i + 1) / batch_size
            bar_length = 50
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            percentage = int(progress * 100)

            elapsed_time = time.time() - start_time
            if elapsed_time > 0.001:
                current_fps = (i + 1) / elapsed_time
                fps_text = f" @ {current_fps:.2f} FPS"
            else:
                fps_text = ""
            sys.stdout.write(f'\r{percentage}% {bar} 100% | Processing frame {i + 1}/{batch_size}{fps_text}')
            sys.stdout.flush()

        if not results_list:
            empty_image_tensor = torch.empty((0, height, width, 3), dtype=torch.float32)
            return {"result": (empty_image_tensor, empty_image_tensor, empty_image_tensor, audio, fps_to_output, all_keypoints_data,)}

        validated_frames = []
        for frame in results_list:
            if frame is None or not isinstance(frame, np.ndarray):
                h, w = image.shape[1], image.shape[2]; frame = np.zeros((h, w, 3), dtype=np.uint8)
            elif frame.ndim == 2: frame = np.stack((frame,) * 3, axis=-1)
            elif frame.shape[2] == 4: frame = frame[:, :, :3]
            elif frame.shape[2] == 1: frame = np.repeat(frame, 3, axis=2)
            if frame.ndim != 3 or frame.shape[2] != 3:
                h, w = image.shape[1], image.shape[2]; frame = np.zeros((h, w, 3), dtype=np.uint8)
            validated_frames.append(frame.astype(np.uint8))
        
        pose_output_frames = []
        blended_output_frames = []
        source_output_frames = []

        for i in range(batch_size):
            original_img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            if original_img_np.shape[2] == 4: original_img_np = original_img_np[:, :, :3]
            elif original_img_np.shape[2] == 1: original_img_np = np.repeat(original_img_np, 3, axis=2)
            
            current_pose_img_np = validated_frames[i]
            pose_output_frames.append(current_pose_img_np)

            resized_pose_for_blend = current_pose_img_np
            if original_img_np.shape != resized_pose_for_blend.shape:
                resized_pose_for_blend = cv2.resize(resized_pose_for_blend, (original_img_np.shape[1], original_img_np.shape[0]))
            blended_img_np = cv2.addWeighted(original_img_np, 0.5, resized_pose_for_blend, 0.5, 0)
            blended_output_frames.append(blended_img_np)

            current_source_img_np = original_img_np.copy()
            source_output_frames.append(current_source_img_np)

        pose_output_tensor = torch.from_numpy(np.array(pose_output_frames).astype(np.float32) / 255.0)
        blended_output_tensor = torch.from_numpy(np.array(blended_output_frames).astype(np.float32) / 255.0)
        source_output_tensor = torch.from_numpy(np.array(source_output_frames).astype(np.float32) / 255.0)
        
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
        return {"ui": ui_output, "result": (pose_output_tensor, blended_output_tensor, source_output_tensor, audio, fps_to_output, all_keypoints_data,)}


NODE_CLASS_MAPPINGS = {
}

NODE_DISPLAY_NAME_MAPPINGS = {
}

if HAS_TENSORRT:
    NODE_CLASS_MAPPINGS["DWposeDeluxeNode"] = DWposeDeluxeNode
    NODE_DISPLAY_NAME_MAPPINGS["DWposeDeluxeNode"] = "DWposeDeluxeNode"
else:
    NODE_CLASS_MAPPINGS["DWposeDeluxeNode"] = DWposeDeluxeNode
    NODE_DISPLAY_NAME_MAPPINGS["DWposeDeluxeNode"] = "DWposeDeluxeNode(CPU only)"