# ComfyUI_DWposeDeluxe/__init__.py

# Handles setup: dependency checks, downloads, folder registration, node import.

from .scripts import logger
import os
import sys
import subprocess
import importlib.util
import folder_paths

try:
    import requests
except ImportError:
    logger.warning(f"'requests' library not found. Attempting to install...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
        import requests
        logger.info(f"'requests' installed successfully.")
    except Exception as e:
        logger.error(f"Failed to install 'requests'. Model downloading may fail\n            {e}")
        requests = None

print(r"""
---------------- Loading DWpose advanced nodes powered by TensorRT ---------------
                                                                            
                    ____  __     __ _____ ____ _____ _____                     
                   ▄▄─▄▄▀█▄─█▀▀▀█─▄█▄─▄▄─█─▄▄─█─▄▄▄▄█▄─▄▄▄▄                    
                   ██─██─██─█─█─█─███─▄▄▄█─██─█▄▄▄▄─██─▄█▀█                    
           ________▀▄▄▄▄▀▀▀▄▄▄▀▄▄▄▀▀▄▄▄▀▀▀▄▄▄▄▀▄▄▄▄▄▀▄▄▄▄▄▀______ ________     
          _____  __/_____ _______ ______________ ___________  __ \___  __/     
         _____  / ___  _ \__  __ \__  ___/_  __ \__  ___/__  /_/ /__  /        
            _  /    /  __/_  / / /_(__  ) / /_/ /_  /  ___  _, _/ _  /         
            /_/     \___/ /_/ /_/ /____/  \____/ /_/     /_/ |_|  /_/          
                                                                            
                                                                            
----------------------- github.com/hobinrude/DWposeDeluxe -----------------------
""")

import importlib.metadata

def is_tensorrt_installed():
    try:
        import tensorrt
        return True, f"[DWposeNode][{fg('green')}INFO{attr('reset')}] TensorRT {tensorrt.__version__}"
    except ImportError:
        for dist in importlib.metadata.distributions():
            if dist.metadata['name'].startswith('tensorrt-cu'):
                return True, f"{dist.metadata['name']} {dist.version}"
    return False, None

HAS_TENSORRT, trt_version_string = is_tensorrt_installed()

if HAS_TENSORRT:
    logger.info(f"{trt_version_string} detected")
else:
    logger.warning(f"TensorRT not found. GPU (TensorRT) inference will not be available")
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            if cuda_version:
                cuda_major_version = "cu" + cuda_version.split('.')[0]
                logger.info(f"Detected CUDA version: {cuda_version}. You may need to install 'tensorrt-{cuda_major_version}' and 'pycuda'")
                logger.info(f"Recommended installation command: pip install tensorrt-{cuda_major_version} pycuda")
                logger.info(f"Installing TensorRT via pip can sometimes cause dependency conflicts with your existing PyTorch installation")
                logger.info(f"It is highly recommended to use a dedicated Python virtual environment for ComfyUI to avoid breaking other installations")
            else:
                logger.warning(f"CUDA is available, but torch.version.cuda is not reporting a version. Cannot recommend specific TensorRT package")
        else:
            logger.warning(f"CUDA is not available. TensorRT is not applicable")
    except ImportError:
        logger.warning(f"PyTorch not found. Cannot check CUDA version for TensorRT recommendations")
    except Exception as e:
        logger.warning(f"An unexpected error occurred while checking for CUDA/TensorRT:\n            {e}")

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
COMFYUI_ROOT = os.path.abspath(os.path.join(NODE_DIR, os.pardir, os.pardir))
BASE_MODEL_DIR = os.path.join(COMFYUI_ROOT, "models", "dwpose")

os.makedirs(BASE_MODEL_DIR, exist_ok=True)

MODEL_URLS = {
    "yolox_l.onnx": ("https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx", BASE_MODEL_DIR),
    "dw-ll_ucoco_384.onnx": ("https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx", BASE_MODEL_DIR),
}

def ensure_models_downloaded():
    if requests is None:
        logger.error(f"Cannot download models because 'requests' library is missing or failed to install")
        return

    logger.info(f"Checking for required ONNX models...")
    download_count = 0
    for model_name, (url, target_dir) in MODEL_URLS.items():
        model_path = os.path.join(target_dir, model_name)
        if not os.path.exists(model_path):
            download_count += 1
            logger.info(f"Downloading {model_name}\n            to {target_dir}...")
            try:
                response = requests.get(url, stream=True, timeout=600)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024 * 1024

                with open(model_path, "wb") as f:
                    downloaded_size = 0
                    for data in response.iter_content(block_size):
                        f.write(data)
                        downloaded_size += len(data)
                        if total_size > 0:
                             done = int(50 * downloaded_size / total_size)
                             sys.stdout.write(f"\r    [{'=' * done}{' ' * (50-done)}] {downloaded_size / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MiB")
                             sys.stdout.flush()
                print()

                if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
                     logger.info(f"{model_name} downloaded successfully")
                else:
                     logger.error(f"Failed to save {model_name} after download")
                     if os.path.exists(model_path): os.remove(model_path)

            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to download {model_name} from {url}\n            {e}")
                if os.path.exists(model_path):
                    try: os.remove(model_path)
                    except OSError: pass
            except Exception as e:
                 logger.error(f"An unexpected error occurred downloading {model_name}\n            {e}")

    if download_count == 0:
        logger.info(f"Required onnx models found in the unified folder")

ensure_models_downloaded()

folder_mappings = {
    "dwpose": (BASE_MODEL_DIR, {".onnx", ".trt"}),
}

logger.info(f"Registering model folders with ComfyUI...")
for key, (path, extensions) in folder_mappings.items():
    if key not in folder_paths.folder_names_and_paths:
        logger.info(f"Registering: {key} -> {path}")
        folder_paths.folder_names_and_paths[key] = ([path], extensions)
    else:
        existing_paths, existing_extensions = folder_paths.folder_names_and_paths[key]
        updated_paths = list(existing_paths)
        if path not in updated_paths:
            updated_paths.append(path)
            folder_paths.folder_names_and_paths[key] = (updated_paths, existing_extensions.union(extensions))
            logger.info(f"Appending path to existing key: {key} -> {path}")

logger.info(f"Importing node definitions...")
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# First, load from the original main.py
try:
    from .main import NODE_CLASS_MAPPINGS as original_mappings, NODE_DISPLAY_NAME_MAPPINGS as original_display_mappings
    NODE_CLASS_MAPPINGS.update(original_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(original_display_mappings)
    logger.info(f"Found {len(original_mappings)} node definition(s) in main.py")
except ImportError as e:
    logger.error(f"Failed to import from main.py: {e}")
except Exception as e:
    logger.error(f"An unexpected error occurred importing from main.py: {e}")

# Then, load from the new nodes/ directory
nodes_dir = os.path.join(os.path.dirname(__file__), "nodes")
if os.path.exists(nodes_dir):
    logger.info(f"Importing node definitions from 'nodes' directory...")
    for filename in os.listdir(nodes_dir):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = filename[:-3]
            filepath = os.path.join(nodes_dir, filename)
            try:
                spec = importlib.util.spec_from_file_location(f"{__name__}.nodes.{module_name}", filepath)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if hasattr(module, "NODE_CLASS_MAPPINGS") and hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
                    logger.info(f"  - Loaded {len(module.NODE_CLASS_MAPPINGS)} node(s) from {filename}")
                else:
                    logger.warning(f"  - Skipping {filename}, does not contain required MAPPINGS.")
            except Exception as e:
                logger.error(f"Failed to import node module from {filepath}: {e}")

WEB_DIRECTORY = "./js"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

from aiohttp import web
from server import PromptServer

@PromptServer.instance.routes.get('/dwpose_adv/get_model_list')
async def get_model_list_api_route(request):
    provider_type = request.query.get('provider_type')
    precision = request.query.get('precision', None)

    from .main import DWposeDeluxeNode
    
    try:
        response_data = DWposeDeluxeNode._get_model_list_api(provider_type, precision)
        return web.json_response(response_data)
    except Exception as e:
        logger.error(f"Failed to get model list\n            {e}")
        return web.json_response({"error": str(e)}, status=500)

@PromptServer.instance.routes.get('/dwpose_adv/get_files')
async def get_files_for_input_dir_route(request):
    try:
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith('.json')]
        if not files:
            files = ["[none]"]
        return web.json_response(sorted(files))
    except Exception as e:
        logger.error(f"Failed to get files for input directory: {e}")
        return web.json_response({"error": str(e)}, status=500)

print("\n-------------------------- DWposeDeluxe nodes loaded -----------------------------")
