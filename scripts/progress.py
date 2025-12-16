# ComfyUI_DWposeDeluxe/scripts/progress.py

from comfy.utils import ProgressBar as UIProgressBar
import sys, time, os, json
from . import memplot

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
except ImportError:
    pynvml = None

# Global Logging State
_LOG_ENABLED = False
_LOG_PATH = None
_LOG_PROVIDER = "CPU"
_NVML_HANDLE = None

def setup(enabled=False, output_dir=None, provider="CPU", batch_size=0, height=0, width=0, poses_to_detect=0):
    global _LOG_ENABLED, _LOG_PATH, _LOG_PROVIDER, _NVML_HANDLE
    
    _LOG_ENABLED = enabled
    _LOG_PROVIDER = provider
    _NVML_HANDLE = None
    _LOG_PATH = None

    if _LOG_ENABLED and output_dir:
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"memlog_{timestamp}.log"
            _LOG_PATH = os.path.join(output_dir, filename)
            
            metadata = {
                'provider': provider,
                'batch_size': batch_size,
                'height': height,
                'width': width,
                'poses_to_detect': poses_to_detect
            }

            # Init NVML
            if _LOG_PROVIDER == "GPU" and pynvml:
                try:
                    pynvml.nvmlInit()
                    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
                except Exception:
                    pass
            
            # Write Headers
            if not os.path.exists(_LOG_PATH):
                with open(_LOG_PATH, "w") as f:
                    meta_str = json.dumps(metadata)
                    f.write(f"# Metadata: {meta_str}\n")
        
        except Exception:
            pass

def finalize():
    global _NVML_HANDLE
    # Shutdown NVML
    if _NVML_HANDLE and pynvml:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    _NVML_HANDLE = None

    # Trigger Plot
    if _LOG_ENABLED and _LOG_PATH and os.path.exists(_LOG_PATH):
        try:
            memplot.create_memplot(_LOG_PATH)
        except Exception as e:
            print(f"[DWposeDLX] Failed to create memory plot: {e}")


FG_FILLED = "\033[38;2;0;100;200m"
FG_EMPTY  = "\033[38;2;0;50;100m"
RESET     = "\033[0m"

class Progress:
    def __init__(self, total, label="Processing", bar_width=50):
        self.total = max(1, int(total))
        self.label = label
        self.bar_width = bar_width
        self.current = 0
        self.start_time = time.time()
        self.ui_bar = UIProgressBar(total)
        self.fps_text = ""  # Store last calculated FPS text
        self.eta_text = ""  # Store last calculated ETA text

        # Write Loop Header if logging is enabled
        if _LOG_ENABLED and _LOG_PATH:
            try:
                ram_total = 0.0
                vram_total = 0.0
                if psutil:
                    ram_total = psutil.virtual_memory().total / (1024**3)
                if _NVML_HANDLE:
                    vram_total = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE).total / (1024**3)

                with open(_LOG_PATH, "a") as f:
                    f.write(f"# [Loop: {self.label}, Total_Steps: {self.total}, RAM_Total: {ram_total:.2f}, VRAM_Total: {vram_total:.2f}]\n")
            except Exception:
                pass

        sys.stdout.write(f'\r  0% {FG_EMPTY}{"░"*self.bar_width}{RESET} 100% | {self.label} 0/{self.total} ')
        sys.stdout.flush()
        time.sleep(0.001)

    def log_memory(self):
        try:
            ram_used = 0.0
            vram_used = 0.0
            
            if psutil:
                ram_used = psutil.virtual_memory().used / (1024**3)
            
            if _NVML_HANDLE:
                vram_used = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE).used / (1024**3)
            
            with open(_LOG_PATH, "a") as f:
                f.write(f"{self.current},{ram_used:.2f},{vram_used:.2f}\n")
        except Exception:
            pass

    def step(self, n=1):
        self.current = min(self.total, self.current + n)
        
        if _LOG_ENABLED and _LOG_PATH:
            self.log_memory()

        progress = self.current / self.total
        filled_len = int(self.bar_width * progress)

        bar = (
            f"{FG_FILLED}{'█' * filled_len}"
            f"{FG_EMPTY}{'░' * (self.bar_width - filled_len)}"
            f"{RESET}"
        )

        percent = int(progress * 100)

        elapsed = max(time.time() - self.start_time, 0.001)
        fps = self.current / elapsed
        self.fps_text = f"{fps:.3f}fps" if self.current > 0 else ""

        remaining = self.total - self.current
        eta = remaining / max(fps, 0.001)
        
        # Convert ETA to MMmSSs format, ensuring consistent "ETA" prefix
        if remaining <= 0: # When loop is complete or nearly complete
            self.eta_text = "ETA 0s"
        elif eta > 60:
            minutes = int(eta // 60)
            seconds = int(eta % 60)
            self.eta_text = f"ETA {minutes}m{seconds:02}s" # :02 pads with leading zero
        else: # eta > 0 and <= 60
            self.eta_text = f"ETA {int(eta)}s"

        sys.stdout.write(
            f'\r{percent:3d}% {bar} 100% | {self.label} {self.current}/{self.total}'
            f' | {self.fps_text} | {self.eta_text} '
        )
        sys.stdout.flush()
        time.sleep(0.001)

        self.ui_bar.update(1)

    def finish(self):
        elapsed_time = time.time() - self.start_time
        
        # Format elapsed time to MMmSSs
        if elapsed_time > 60:
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            elapsed_text = f"{minutes}m{seconds:02}s "
        else:
            elapsed_text = f"{int(elapsed_time)}s "

        # Overwrite the last progress bar line with final 100%, FPS and ELP time
        final_bar = (
            f"{FG_FILLED}{'█' * self.bar_width}"
            f"{FG_EMPTY}{'░' * (0)}"
            f"{RESET}"
        )
        sys.stdout.write(
            f'\r100% {final_bar} 100% | {self.label} {self.total}/{self.total}'
            f' | {self.fps_text} | ELP {elapsed_text} ' # Display FPS and ELP time
        )
        sys.stdout.write("\n")
        sys.stdout.flush()


def progress(total, label="Processing", bar_width=60):
    return Progress(total, label, bar_width)

progress.setup = setup
progress.finalize = finalize