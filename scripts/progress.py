from comfy.utils import ProgressBar as UIProgressBar
import sys, time

FG_FILLED = "\033[38;2;0;50;100m"
FG_EMPTY  = "\033[38;2;0;50;100m"
RESET     = "\033[0m"

class Progress:
    def __init__(self, total, label="Processing", bar_width=50):
        self.total = max(1, int(total))
        self.label = label
        self.bar_width = bar_width
        self.current = 0
        self.start_time = time.time()
        self.ui_bar = UIProgressBar(total)  # native ComfyUI progress

        sys.stdout.write(f'\r  0% {FG_EMPTY}{"░"*self.bar_width}{RESET} 100% | {self.label} 0/{self.total} ')
        sys.stdout.flush()
        time.sleep(0.001)

    def step(self, n=1):
        self.current = min(self.total, self.current + n)

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
        fps_text = f"{fps:.3f}fps" if self.current > 0 else ""

        # ETA (seconds remaining)
        remaining = self.total - self.current
        eta = remaining / max(fps, 0.001)
        eta_text = f"ETA {eta:.1f}s" if self.current > 0 else ""

        sys.stdout.write(
            f'\r{percent:3d}% {bar} 100% | {self.label} {self.current}/{self.total}'
            f' | {fps_text} | {eta_text} '
        )
        sys.stdout.flush()
        time.sleep(0.001)

        self.ui_bar.update(1)

    def finish(self):
        sys.stdout.write("\n")
        sys.stdout.flush()


def progress(total, label="Processing", bar_width=60):
    return Progress(total, label, bar_width)
