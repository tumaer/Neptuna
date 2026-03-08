import torch
import numpy as np
import os
import socket
import time
import threading
import subprocess
import shutil
import re
import resource
from collections import defaultdict
from typing import List, Dict, Optional
import torch.distributed as dist
import json
from zoneinfo import ZoneInfo


def detect_runtime_backend() -> str:
    """Detect the active accelerator backend.

    Returns:
        One of: ``"cuda"``, ``"xpu"``, ``"mps"``, or ``"cpu"``.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_rank_world() -> tuple[int, int]:
    """Return ``(rank, world_size)`` for the current process.

    Behavior:
        1. If ``torch.distributed`` is initialized, use that source of truth.
        2. Otherwise, fall back to ``RANK``/``WORLD_SIZE`` environment variables.
        3. If rank is missing, default to rank 0 and world size 1.

    Returns:
        Tuple containing the process rank and total world size.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    rank = int(os.environ.get("RANK", -1))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if rank < 0:
        rank = 0
    return rank, world


def _sharded_local_count(global_count: int, rank: int, world_size: int) -> int:
    """Compute deterministic per-rank shard size for a global sample count.

    Remainder samples are assigned to lower ranks first, matching the typical
    contiguous sharding strategy.
    """
    if world_size <= 1:
        return int(global_count)
    base = int(global_count) // world_size
    rem = int(global_count) % world_size
    return base + (1 if rank < rem else 0)


def _as_float_or_none(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _mean_std(values: List[float]) -> tuple[Optional[float], Optional[float]]:
    arr = [float(v) for v in values if v is not None]
    if len(arr) == 0:
        return None, None
    if len(arr) == 1:
        return arr[0], 0.0
    np_arr = np.asarray(arr, dtype=np.float64)
    return float(np_arr.mean()), float(np_arr.std(ddof=0))


def now_berlin_iso() -> str:
    """Return current wall-clock time in Europe/Berlin as ISO-8601."""
    return datetime_now_berlin().isoformat()


def datetime_now_berlin():
    """Return timezone-aware datetime in Europe/Berlin."""
    from datetime import datetime
    return datetime.now(ZoneInfo("Europe/Berlin"))


def round_nested_numbers(obj, digits: int = 2):
    """Recursively round floating-point values in nested containers."""
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), digits)
    if isinstance(obj, dict):
        return {k: round_nested_numbers(v, digits=digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_nested_numbers(v, digits=digits) for v in obj]
    if isinstance(obj, tuple):
        return tuple(round_nested_numbers(v, digits=digits) for v in obj)
    return obj


def _overall_mean_std_from_device_stats(
    device_stats: List[Dict],
    mean_key: str,
    std_key: str,
    count_key: str,
) -> tuple[Optional[float], Optional[float]]:
    """Compute pooled mean/std across devices using per-device mean/std/count.

    Uses population variance (ddof=0):
        var = (sum_i n_i (s_i^2 + (m_i - m)^2)) / sum_i n_i
    where m_i/s_i/n_i are per-device mean/std/count.
    """
    weighted_count = 0.0
    weighted_mean_sum = 0.0
    prepared = []
    for stats in device_stats:
        n = _as_float_or_none(stats.get(count_key))
        m = _as_float_or_none(stats.get(mean_key))
        s = _as_float_or_none(stats.get(std_key))
        if n is None or n <= 0 or m is None:
            continue
        if s is None:
            s = 0.0
        n = float(n)
        m = float(m)
        s = float(s)
        prepared.append((n, m, s))
        weighted_count += n
        weighted_mean_sum += n * m

    if weighted_count <= 0:
        return None, None

    mean = weighted_mean_sum / weighted_count
    var_num = 0.0
    for n, m, s in prepared:
        var_num += n * ((s ** 2) + ((m - mean) ** 2))
    var = max(0.0, var_num / weighted_count)
    return float(mean), float(np.sqrt(var))


class RuntimeTelemetryScope:
    """
        Runtime monitor for a scoped inference phase.

        The scope is used as a context manager around inference
        sections (for example a single `trainer.predict(...)` call).

        Capabilities:
            - Tracks elapsed wall time.
            - Captures peak process memory and accelerator memory where available.
            - Samples device utilization/power with best-effort backend-specific
                sources (e.g., `nvidia-smi`, `xpu-smi`, `xpumcli`, `powermetrics`).
            - Produces a rank-local structured report consumable by
                `aggregate_runtime_report()`.
    """

    def __init__(self, name: str, sample_interval_sec: float = 5.0):
        # Human-readable scope label (for example: "eval_random_start").
        self.name = name

        # Keep a lower bound to avoid very aggressive polling.
        self.sample_interval_sec = max(0.01, float(sample_interval_sec))

        # Snapshot runtime context once at construction.
        self.backend = detect_runtime_backend()
        self.hostname = socket.gethostname()
        self.rank, self.world_size = get_rank_world()

        self._start_ts = None
        self._end_ts = None
        self._running = False
        self._thread = None
        self._stop_event = threading.Event()

        # Per-device rolling sample buffers.
        self._gpu_samples = defaultdict(lambda: {"util": [], "power_w": []})
        self._mps_memory_samples_gb = []
        self._sample_loop_iterations = 0

        # Internal provenance/debug notes. Kept internal unless explicitly
        # surfaced by callers.
        self._telemetry_info = {
            "utilization_source": None,
            "power_source": None,
            "notes": [],
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def _active_device_labels(self) -> List[str]:
        """Return display labels for devices associated with this process.

        In distributed mode we report the current rank-local device; otherwise
        we list all visible devices for the backend.
        """
        if self.backend == "cuda":
            try:
                if self.world_size > 1:
                    idx = int(torch.cuda.current_device())
                    name = torch.cuda.get_device_name(idx)
                    return [f"cuda:{idx} ({name})"]
                labels = []
                for idx in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(idx)
                    labels.append(f"cuda:{idx} ({name})")
                return labels
            except Exception:
                return ["cuda"]

        if self.backend == "xpu":
            try:
                if self.world_size > 1:
                    idx = int(torch.xpu.current_device())
                    name = torch.xpu.get_device_name(idx)
                    return [f"xpu:{idx} ({name})"]
                labels = []
                for idx in range(torch.xpu.device_count()):
                    name = torch.xpu.get_device_name(idx)
                    labels.append(f"xpu:{idx} ({name})")
                return labels
            except Exception:
                return ["xpu"]

        if self.backend == "mps":
            return ["mps:0"]
        return ["cpu"]

    def start(self):
        """Start timing and background sampling for this scope.

        This method is idempotent for already-running scopes.
        """
        if self._running:
            return
        self._running = True
        self._start_ts = time.perf_counter()

        # Reset accelerator peak-memory counters at scope start so reported
        # peaks are local to this scope window.
        if self.backend == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        elif self.backend == "xpu":
            try:
                if hasattr(torch.xpu, "reset_peak_memory_stats"):
                    torch.xpu.reset_peak_memory_stats()
            except Exception:
                pass

        self._stop_event.clear()
        # Background sampler keeps runtime overhead low and decoupled from the
        # model forward path.
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop sampling and finalize elapsed timing for this scope.

        Safe to call multiple times; only the first call while running has
        effect.
        """
        if not self._running:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2.0 * self.sample_interval_sec))
        self._end_ts = time.perf_counter()
        self._running = False

    @property
    def elapsed_sec(self) -> float:
        """Return elapsed scope duration in seconds.

        If the scope is still running, this is computed up to "now".
        """
        end_ts = self._end_ts if self._end_ts is not None else time.perf_counter()
        start_ts = self._start_ts if self._start_ts is not None else end_ts
        return max(0.0, end_ts - start_ts)

    def _sample_loop(self):
        # Run until `stop()` signals termination.
        while not self._stop_event.is_set():
            self._sample_once()
            self._stop_event.wait(self.sample_interval_sec)

    def _sample_once(self):
        self._sample_loop_iterations += 1
        # Backend-specific sampling is intentionally split into dedicated
        # methods to isolate vendor/tooling differences.
        if self.backend == "cuda":
            self._sample_cuda_util_power()
        elif self.backend == "xpu":
            self._sample_xpu_util_power()
        elif self.backend == "mps":
            self._sample_mps_memory()
            self._sample_mps_util_power()

    def _sample_mps_memory(self):
        try:
            if hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
                mem_gb = float(torch.mps.current_allocated_memory()) / (1024.0 ** 3)
                self._mps_memory_samples_gb.append(mem_gb)
        except Exception:
            pass

    def _sample_cuda_util_power(self):
        # CUDA telemetry is sourced from nvidia-smi for broad compatibility.
        target_selectors = self._target_cuda_gpu_selectors()
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,uuid,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=2.0)
        except Exception:
            return

        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            gpu_id = parts[0]
            gpu_uuid = parts[1]

            # In multi-process runs, keep telemetry rank-local so each rank
            # reports only the accelerator it is actually using.
            if target_selectors is not None:
                target_ids = target_selectors.get("ids", set())
                target_uuids = target_selectors.get("uuids", set())
                id_match = bool(target_ids) and (gpu_id in target_ids)
                uuid_match = bool(target_uuids) and (gpu_uuid.lower() in target_uuids)
                if not (id_match or uuid_match):
                    continue

            try:
                util = float(parts[2]) if parts[2] not in ("", "N/A") else None
                pwr = float(parts[3]) if parts[3] not in ("", "N/A") else None
            except Exception:
                util = None
                pwr = None
            if util is not None:
                self._gpu_samples[gpu_id]["util"].append(util)
            if pwr is not None:
                self._gpu_samples[gpu_id]["power_w"].append(pwr)

        self._telemetry_info["utilization_source"] = "nvidia-smi"
        self._telemetry_info["power_source"] = "nvidia-smi"

    def _target_cuda_gpu_selectors(self) -> Optional[Dict[str, set[str]]]:
        """Return rank-local CUDA selectors for nvidia-smi rows.

        Behavior:
        - Single-process: sample all GPUs (returns ``None``).
        - Multi-process: sample only rank-local GPU (by index or UUID).

        We map logical ``torch.cuda.current_device()`` through
        ``CUDA_VISIBLE_DEVICES`` when available. Tokens may be integer indices
        or GPU UUIDs depending on launcher/scheduler setup.
        """
        if self.backend != "cuda" or self.world_size <= 1:
            return None

        try:
            logical_idx = int(torch.cuda.current_device())
        except Exception:
            return None

        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not cvd:
            # No explicit masking: logical index usually matches physical index.
            return {"ids": {str(logical_idx)}, "uuids": set()}

        tokens = [t.strip() for t in cvd.split(",") if t.strip()]
        if not (0 <= logical_idx < len(tokens)):
            return {"ids": {str(logical_idx)}, "uuids": set()}

        selected = tokens[logical_idx]
        if re.fullmatch(r"\d+", selected):
            return {"ids": {str(int(selected))}, "uuids": set()}

        # UUID token path (e.g. GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx).
        return {"ids": set(), "uuids": {selected.lower()}}

    def _append_note_once(self, message: str):
        if message not in self._telemetry_info["notes"]:
            self._telemetry_info["notes"].append(message)

    def _sample_xpu_util_power(self):
        # 1) Try PyTorch-native utilization if available.
        try:
            if hasattr(torch, "xpu") and hasattr(torch.xpu, "utilization"):
                if self.world_size > 1:
                    dev_ids = [int(torch.xpu.current_device())]
                else:
                    dev_ids = list(range(int(torch.xpu.device_count())))
                for dev_id in dev_ids:
                    try:
                        util = torch.xpu.utilization(dev_id)
                        if util is not None:
                            self._gpu_samples[str(dev_id)]["util"].append(float(util))
                    except Exception:
                        pass
                self._telemetry_info["utilization_source"] = "torch.xpu.utilization"
        except Exception:
            pass

        # 2) Try xpu-smi if installed. Expected csv-like output lines:
        #    <id>,<util>,<power>
        if shutil.which("xpu-smi") is not None:
            cmd = [
                "xpu-smi",
                "stats",
                "-d",
                "all",
                "--format",
                "csv",
                "--show",
                "utilization,power",
            ]
            try:
                out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=2.0)
                parsed_any = False
                for line in out.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 3:
                        continue
                    dev = parts[0]
                    try:
                        util = float(parts[1]) if parts[1] not in ("", "N/A", "na") else None
                        pwr = float(parts[2]) if parts[2] not in ("", "N/A", "na") else None
                    except Exception:
                        util, pwr = None, None
                    if util is not None:
                        self._gpu_samples[str(dev)]["util"].append(util)
                        parsed_any = True
                    if pwr is not None:
                        self._gpu_samples[str(dev)]["power_w"].append(pwr)
                        parsed_any = True
                if parsed_any:
                    self._telemetry_info["utilization_source"] = self._telemetry_info["utilization_source"] or "xpu-smi"
                    self._telemetry_info["power_source"] = "xpu-smi"
                    return
            except Exception:
                pass

        # 3) Try xpumcli (JSON/text). Parse best-effort util/power pairs.
        if shutil.which("xpumcli") is not None:
            # NOTE: xpumcli format varies by version; we do regex extraction.
            cmd = ["xpumcli", "stats", "-d", "-1"]
            try:
                out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=2.0)
                # A loose parser: look for device id and nearby util/power numbers.
                # We avoid hard dependency on a specific xpumcli schema.
                current_dev = None
                parsed_any = False
                for raw in out.splitlines():
                    line = raw.strip()
                    m_dev = re.search(r"device\s*id\s*[:=]\s*(\d+)", line, flags=re.IGNORECASE)
                    if m_dev:
                        current_dev = m_dev.group(1)
                        continue

                    m_util = re.search(r"(gpu\s*util\w*|utili[sz]ation)\D+([0-9]+(?:\.[0-9]+)?)", line, flags=re.IGNORECASE)
                    if m_util and current_dev is not None:
                        self._gpu_samples[str(current_dev)]["util"].append(float(m_util.group(2)))
                        parsed_any = True

                    m_pwr = re.search(r"power\D+([0-9]+(?:\.[0-9]+)?)", line, flags=re.IGNORECASE)
                    if m_pwr and current_dev is not None:
                        self._gpu_samples[str(current_dev)]["power_w"].append(float(m_pwr.group(1)))
                        parsed_any = True

                if parsed_any:
                    self._telemetry_info["utilization_source"] = self._telemetry_info["utilization_source"] or "xpumcli"
                    self._telemetry_info["power_source"] = self._telemetry_info["power_source"] or "xpumcli"
                    return
            except Exception:
                pass

        # If we get here, no xpu util/power source succeeded this sample.
        if self._telemetry_info["utilization_source"] is None:
            self._append_note_once("XPU utilization telemetry unavailable (torch.xpu.utilization/xpu-smi/xpumcli not usable).")
        if self._telemetry_info["power_source"] is None:
            self._append_note_once("XPU power telemetry unavailable (xpu-smi/xpumcli not usable).")

    def _sample_mps_util_power(self):
        # There is no stable PyTorch MPS API for utilization/power. Try powermetrics best-effort.
        if shutil.which("powermetrics") is None:
            self._append_note_once("MPS utilization/power telemetry unavailable (powermetrics not found).")
            return

        # `powermetrics` often needs elevated privileges; keep this optional/non-fatal.
        cmd = ["powermetrics", "--samplers", "gpu_power", "-n", "1", "-i", "1000"]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=3.0)
        except Exception:
            self._append_note_once("powermetrics exists but could not be executed (permission or platform constraints).")
            return

        util = None
        pwr = None
        for raw in out.splitlines():
            line = raw.strip()
            # Best-effort parsing across macOS versions
            m_util = re.search(r"(gpu\s*active|utili[sz]ation)\D+([0-9]+(?:\.[0-9]+)?)\s*%", line, flags=re.IGNORECASE)
            if m_util:
                util = float(m_util.group(2))
            m_pwr = re.search(r"gpu\s*power\D+([0-9]+(?:\.[0-9]+)?)\s*(w|mw)", line, flags=re.IGNORECASE)
            if m_pwr:
                val = float(m_pwr.group(1))
                unit = m_pwr.group(2).lower()
                pwr = val / 1000.0 if unit == "mw" else val

        dev = "0"
        if util is not None:
            self._gpu_samples[dev]["util"].append(util)
            self._telemetry_info["utilization_source"] = "powermetrics"
        if pwr is not None:
            self._gpu_samples[dev]["power_w"].append(pwr)
            self._telemetry_info["power_source"] = "powermetrics"

        if util is None and pwr is None:
            self._append_note_once("powermetrics output did not include parseable MPS util/power samples.")

    def _peak_memory_info(self) -> Dict[str, Optional[float]]:
        # ru_maxrss on Linux is KB; convert to GB for consistency with
        # accelerator memory reporting.
        cpu_peak_gb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 ** 2)
        info: Dict[str, Optional[float]] = {
            "cpu_peak_rss_gb": cpu_peak_gb,
            "accelerator_peak_allocated_gb": None,
            "accelerator_peak_reserved_gb": None,
        }

        if self.backend == "cuda":
            try:
                dev = torch.cuda.current_device()
                info["accelerator_peak_allocated_gb"] = float(torch.cuda.max_memory_allocated(dev)) / (1024.0 ** 3)
                info["accelerator_peak_reserved_gb"] = float(torch.cuda.max_memory_reserved(dev)) / (1024.0 ** 3)
            except Exception:
                pass
        elif self.backend == "xpu":
            try:
                dev = torch.xpu.current_device()
                if hasattr(torch.xpu, "max_memory_allocated"):
                    info["accelerator_peak_allocated_gb"] = float(torch.xpu.max_memory_allocated(dev)) / (1024.0 ** 3)
                if hasattr(torch.xpu, "max_memory_reserved"):
                    info["accelerator_peak_reserved_gb"] = float(torch.xpu.max_memory_reserved(dev)) / (1024.0 ** 3)
            except Exception:
                pass
        elif self.backend == "mps":
            if self._mps_memory_samples_gb:
                info["accelerator_peak_allocated_gb"] = float(max(self._mps_memory_samples_gb))

        return info

    def build_local_report(self, local_samples: int) -> Dict:
        """Build a rank-local runtime summary payload.

        Args:
            local_samples: Number of samples processed by this rank within the
                monitored scope.

        Returns:
            A JSON-serializable dictionary containing elapsed time, throughput,
            memory peaks, per-device telemetry, and telemetry provenance.
        """
        # Protect throughput division against near-zero elapsed values.
        elapsed_sec = max(self.elapsed_sec, 1e-12)
        elapsed_min = elapsed_sec / 60.0
        throughput = float(local_samples) / elapsed_sec

        per_device = {}
        for gpu_id, gpu_series in self._gpu_samples.items():
            util_mean, util_std = _mean_std(gpu_series.get("util", []))
            pwr_mean, pwr_std = _mean_std(gpu_series.get("power_w", []))
            per_device[str(gpu_id)] = {
                "utilization_mean_percent": util_mean,
                "utilization_std_percent": util_std,
                "power_mean_w": pwr_mean,
                "power_std_w": pwr_std,
            }

        report = {
            "scope_name": self.name,
            "rank": self.rank,
            "world_size": self.world_size,
            "hostname": self.hostname,
            "backend": self.backend,
            "device_labels": self._active_device_labels(),
            "elapsed_min": float(elapsed_min),
            "local_samples": int(local_samples),
            "throughput_samples_per_sec": float(throughput),
            "telemetry_sampling": {
                "sample_interval_sec": float(self.sample_interval_sec),
                "samples_logged_total": int(self._sample_loop_iterations),
            },
            "memory": self._peak_memory_info(),
            "device_telemetry": {
                "per_device": per_device,
            },
        }
        return report


def aggregate_runtime_report(local_report: Dict, global_samples: Optional[int] = None) -> Dict:
    """Aggregate rank-local scope reports into a world-level report.

    In distributed runs, this function gathers local reports from all ranks via
    ``dist.all_gather_object``. In single-process mode, the output is built from
    the provided local report only.

    Args:
        local_report: Rank-local report from
            ``RuntimeTelemetryScope.build_local_report()``.
        global_samples: Optional known global sample count. If not provided,
            the sum of local sample counts is used.

    Returns:
        A structured aggregate containing elapsed scope time, global/per-device
        throughput, telemetry availability, source metadata, and rank reports.
    """
    rank, world = get_rank_world()

    # In distributed mode, every rank contributes one local report.
    reports = [local_report]
    if dist.is_available() and dist.is_initialized() and world > 1:
        gathered = [None for _ in range(world)]
        dist.all_gather_object(gathered, local_report)
        reports = [r for r in gathered if isinstance(r, dict)]

    elapsed_values = [
        _as_float_or_none(r.get("elapsed_min"))
        for r in reports
    ]
    elapsed_values = [v for v in elapsed_values if v is not None]
    # Scope duration is the max rank duration so global throughput reflects
    # end-to-end wall time.
    elapsed_scope = max(elapsed_values) if elapsed_values else None

    per_device_throughput = [
        _as_float_or_none(r.get("throughput_samples_per_sec"))
        for r in reports
    ]
    per_device_throughput = [v for v in per_device_throughput if v is not None]
    thr_mean, thr_std = _mean_std(per_device_throughput)

    per_device_gpu_stats = []
    util_agg_stats = []
    power_agg_stats = []
    for r in reports:
        r_rank = r.get("rank")
        host = r.get("hostname")
        # New unified schema with backward-compatible fallback.
        local_device_telemetry = r.get("device_telemetry") or {}
        per_local_device = local_device_telemetry.get("per_device") or r.get("gpu_devices") or {}
        local_device_count = max(1, int(len(per_local_device)))
        samples_logged_total = _as_float_or_none((r.get("telemetry_sampling") or {}).get("samples_logged_total"))
        inferred_count_per_device = None
        if samples_logged_total is not None and samples_logged_total > 0:
            inferred_count_per_device = float(samples_logged_total) / float(local_device_count)
        for gpu_id, stats in per_local_device.items():
            per_device_gpu_stats.append(
                {
                    "rank": r_rank,
                    "hostname": host,
                    "gpu_id": gpu_id,
                    **stats,
                }
            )

            util_mean = _as_float_or_none(stats.get("utilization_mean_percent"))
            util_std = _as_float_or_none(stats.get("utilization_std_percent"))
            util_n = _as_float_or_none(stats.get("utilization_sample_count"))
            if util_n is None:
                util_n = inferred_count_per_device
            util_agg_stats.append(
                {
                    "utilization_mean_percent": util_mean,
                    "utilization_std_percent": util_std,
                    "sample_count": util_n,
                }
            )

            pwr_mean = _as_float_or_none(stats.get("power_mean_w"))
            pwr_std = _as_float_or_none(stats.get("power_std_w"))
            pwr_n = _as_float_or_none(stats.get("power_sample_count"))
            if pwr_n is None:
                pwr_n = inferred_count_per_device
            power_agg_stats.append(
                {
                    "power_mean_w": pwr_mean,
                    "power_std_w": pwr_std,
                    "sample_count": pwr_n,
                }
            )

    util_mean, util_std = _overall_mean_std_from_device_stats(
        util_agg_stats,
        mean_key="utilization_mean_percent",
        std_key="utilization_std_percent",
        count_key="sample_count",
    )
    power_mean, power_std = _overall_mean_std_from_device_stats(
        power_agg_stats,
        mean_key="power_mean_w",
        std_key="power_std_w",
        count_key="sample_count",
    )

    # Summed local counts are used by default unless caller provides an
    # authoritative global sample count.
    total_local_samples = int(sum(int(r.get("local_samples", 0)) for r in reports))
    total_samples = int(global_samples) if global_samples is not None else total_local_samples
    global_throughput = None
    if elapsed_scope is not None and elapsed_scope > 0:
        global_throughput = float(total_samples) / float(elapsed_scope * 60.0)

    peak_cpu_gb = [
        _as_float_or_none((r.get("memory") or {}).get("cpu_peak_rss_gb"))
        for r in reports
    ]
    peak_accel_gb = [
        _as_float_or_none((r.get("memory") or {}).get("accelerator_peak_allocated_gb"))
        for r in reports
    ]

    sample_intervals = [
        _as_float_or_none((r.get("telemetry_sampling") or {}).get("sample_interval_sec"))
        for r in reports
    ]
    sample_intervals = [v for v in sample_intervals if v is not None and v > 0]
    # All ranks are expected to share the same configured interval; choose first.
    sample_interval_sec = sample_intervals[0] if sample_intervals else None

    total_samples_logged = int(sum(int((r.get("telemetry_sampling") or {}).get("samples_logged_total", 0) or 0) for r in reports))
    aggregated = {
        "scope_name": local_report.get("scope_name", "scope"),
        "elapsed_min": elapsed_scope,
        "rank": rank,
        "world_size": world,
        "backend": local_report.get("backend"),
        "hostnames": sorted(list({str(r.get("hostname")) for r in reports if r.get("hostname")})),
        "devices": {
            "devices_reporting": int(len(reports)),
            "accelerator_devices_reporting": int(len(per_device_gpu_stats)),
        },
        "telemetry_sampling": {
            "sample_interval_sec": sample_interval_sec,
            "samples_logged_total": total_samples_logged,
        },
        "samples": {
            "global_samples": total_samples,
            "summed_local_samples": total_local_samples,
        },
        "throughput": {
            "global_samples_per_sec": global_throughput,
            "per_device_samples_per_sec": per_device_throughput,
            "per_device_mean_samples_per_sec": thr_mean,
            "per_device_std_samples_per_sec": thr_std,
        },
        "device_telemetry": {
            "per_device": per_device_gpu_stats,
            "overall_utilization": {
                "mean_percent": util_mean,
                "std_percent": util_std,
            },
            "overall_power": {
                "mean_w": power_mean,
                "std_w": power_std,
            },
        },
        "peak_memory": {
            "cpu_peak_rss_gb_max": float(max([v for v in peak_cpu_gb if v is not None])) if any(v is not None for v in peak_cpu_gb) else None,
            "accelerator_peak_allocated_gb_max": float(max([v for v in peak_accel_gb if v is not None])) if any(v is not None for v in peak_accel_gb) else None,
        },
    }
    return aggregated


def write_runtime_log(log_path: str, payload: Dict) -> None:
    """Atomically write inference runtime telemetry to JSON.

    The payload is written to a temporary file first and then moved into place,
    reducing the chance of partially written logs.

    Args:
        log_path: Final destination JSON path.
        payload: JSON-serializable runtime report payload.
    """
    # Atomic replace prevents consumers from seeing partially-written JSON.
    tmp_path = f"{log_path}.tmp"
    payload_rounded = round_nested_numbers(payload, digits=2)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload_rounded, f, indent=2, sort_keys=False)
    os.replace(tmp_path, log_path)


def estimate_local_sample_count(predictions_obj, global_dataset_len: int) -> int:
    """Estimate per-rank processed sample count from predict output.

    This helper handles cases where prediction tensors may be globally gathered
    (and thus larger than expected for one rank). In those cases it falls back
    to an expected shard size derived from ``global_dataset_len``.

    Args:
        predictions_obj: Prediction object (typically from `trainer.predict`) 
            expected to expose ``predictions``.
        global_dataset_len: Global dataset length used for inference.

    Returns:
        Estimated local sample count for the current rank.
    """
    rank, world = get_rank_world()
    n_local = None
    try:
        preds = getattr(predictions_obj, "predictions", None)
        if hasattr(preds, "shape") and len(preds.shape) > 0:
            n_local = int(preds.shape[0])
    except Exception:
        n_local = None

    # Expected local shard size from global dataset metadata.
    fallback = _sharded_local_count(global_dataset_len, rank, world)
    if n_local is None or n_local <= 0:
        return fallback

    # If predict() returns globally gathered tensors per rank, this value can be too large.
    # In that case, fall back to the expected sharded count.
    if world > 1 and n_local > global_dataset_len:
        return fallback
    return n_local