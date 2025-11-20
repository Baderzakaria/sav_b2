import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional

from collections import deque

try:
    import pynvml
except ImportError:  # pragma: no cover - handled at runtime
    pynvml = None


@dataclass
class GPUMetric:
    timestamp: float
    gpu_util: float
    mem_used_mb: float
    mem_total_mb: float
    temperature: float

    @property
    def mem_free_mb(self) -> float:
        return max(self.mem_total_mb - self.mem_used_mb, 0)


@dataclass
class GPUWatcher:
    gpu_index: int = 0
    mem_floor_mb: int = 2048
    temp_ceiling_c: int = 80
    poll_interval_sec: float = 2.0
    max_history: int = 300
    num_workers: int = 1
    _history: Deque[GPUMetric] = field(default_factory=deque, init=False)
    _threads: List[threading.Thread] = field(default_factory=list, init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _alert_callback: Optional[Callable[[str, GPUMetric], None]] = field(default=None, init=False)

    def start(self, alert_callback: Optional[Callable[[str, GPUMetric], None]] = None) -> None:
        if pynvml is None:
            raise RuntimeError("pynvml is required for GPU monitoring. Install with `pip install pynvml`.")
        if any(thread.is_alive() for thread in self._threads):
            return
        pynvml.nvmlInit()
        self._alert_callback = alert_callback
        self._stop_event.clear()
        self._threads = []
        worker_count = max(1, self.num_workers)
        for _ in range(worker_count):
            thread = threading.Thread(target=self._poll_loop, daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=2)
        if pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def _poll_loop(self) -> None:
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        while not self._stop_event.is_set():
            timestamp = time.time()
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            metric = GPUMetric(
                timestamp=timestamp,
                gpu_util=float(util.gpu),
                mem_used_mb=mem.used / (1024 ** 2),
                mem_total_mb=mem.total / (1024 ** 2),
                temperature=float(temp),
            )
            with self._lock:
                self._history.append(metric)
                while len(self._history) > self.max_history:
                    self._history.popleft()
            self._evaluate_thresholds(metric)
            time.sleep(self.poll_interval_sec)

    def _evaluate_thresholds(self, metric: GPUMetric) -> None:
        if not self._alert_callback:
            return
        if metric.mem_free_mb < self.mem_floor_mb:
            self._alert_callback("memory_floor", metric)
        if metric.temperature > self.temp_ceiling_c:
            self._alert_callback("temp_ceiling", metric)

    def history(self) -> List[Dict[str, float]]:
        with self._lock:
            return [
                {
                    "timestamp": m.timestamp,
                    "gpu_util": m.gpu_util,
                    "mem_used_mb": m.mem_used_mb,
                    "mem_total_mb": m.mem_total_mb,
                    "temperature": m.temperature,
                }
                for m in list(self._history)
            ]

    def latest(self) -> Optional[Dict[str, float]]:
        with self._lock:
            if not self._history:
                return None
            m = self._history[-1]
            return {
                "timestamp": m.timestamp,
                "gpu_util": m.gpu_util,
                "mem_used_mb": m.mem_used_mb,
                "mem_total_mb": m.mem_total_mb,
                "temperature": m.temperature,
            }


def capture_nvidia_smi() -> str:
    """Return the current `nvidia-smi` output as text."""
    try:
        return subprocess.check_output(["nvidia-smi"], text=True, timeout=5)
    except Exception as exc:  # pragma: no cover
        return f"Failed to run nvidia-smi: {exc}"



