import time
import threading
import psutil

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ResourceMonitor:
    def __init__(self, interval=0.1):
        """
        interval: measurement interval in seconds.
        0.1 = measure every 100 ms.
        """
        self.interval = interval

        self.running = False
        self.thread = None

        self.ram_samples = []
        self.vram_samples = []
        self.cpu_samples = []

    def _get_ram_mb(self):
        """Get current RAM usage of this Python process."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 2)

    def _get_vram_mb(self):
        """Get current VRAM usage."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return 0.0

        return torch.cuda.memory_allocated() / (1024 ** 2)

    def _monitor(self):
        """Continuously collect resource measurements."""

        while self.running:

            # RAM
            ram = self._get_ram_mb()
            self.ram_samples.append(ram)

            # VRAM
            vram = self._get_vram_mb()
            self.vram_samples.append(vram)

            # CPU
            cpu = psutil.cpu_percent(interval=None)
            self.cpu_samples.append(cpu)

            time.sleep(self.interval)

    def start(self):
        """Start resource monitoring."""

        # Reset previous measurements
        self.ram_samples = []
        self.vram_samples = []
        self.cpu_samples = []

        # Reset PyTorch peak memory statistics
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Initialize CPU measurement
        psutil.cpu_percent(interval=None)

        self.running = True

        self.thread = threading.Thread(
            target=self._monitor,
            daemon=True
        )

        self.thread.start()

    def stop(self):
        """Stop resource monitoring."""

        self.running = False

        if self.thread is not None:
            self.thread.join()

    def get_results(self):
        """Return calculated resource metrics."""

        if not self.ram_samples:
            return {
                "avg_ram_mb": 0,
                "peak_ram_mb": 0,
                "avg_vram_mb": 0,
                "peak_vram_mb": 0,
                "avg_cpu_percent": 0,
                "peak_cpu_percent": 0
            }

        return {
            "avg_ram_mb": sum(self.ram_samples) / len(self.ram_samples),
            "peak_ram_mb": max(self.ram_samples),

            "avg_vram_mb": (
                sum(self.vram_samples) / len(self.vram_samples)
                if self.vram_samples
                else 0
            ),

            "peak_vram_mb": (
                max(self.vram_samples)
                if self.vram_samples
                else 0
            ),

            "avg_cpu_percent": (
                sum(self.cpu_samples) / len(self.cpu_samples)
                if self.cpu_samples
                else 0
            ),

            "peak_cpu_percent": (
                max(self.cpu_samples)
                if self.cpu_samples
                else 0
            )
        }


def measure_operation(operation):
    """
        Measure resources while executing an operation.
        operation must be a function.
        results = func()

        result, metrics = measure_operation(
                lambda: func()
            )
    """

    monitor = ResourceMonitor(interval=0.1)

    monitor.start()

    try:
        result = operation()
    finally:
        monitor.stop()

    metrics = monitor.get_results()

    return result, metrics