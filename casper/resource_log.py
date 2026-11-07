from __future__ import annotations

import resource
import time
from contextlib import contextmanager

import torch

# The per-epoch cost fields every trainer logs, with the meaning the MNIST run JSON gives them.
RESOURCE_FIELDS = ["epoch_time_s", "train_time_s", "eval_time_s", "s_per_train_example",
                   "gpu_mem_peak_mb", "gpu_mem_reserved_peak_mb", "cpu_rss_peak_mb"]


class EpochResources:
    """Wall time and peak memory of one epoch, measured the same way by every trainer.

    Call start() at the beginning of an epoch, wrap training and evaluation in
    phase("train") / phase("eval"), and read row() at the end. Times are
    synchronised with CUDA; GPU peaks are reset at start() so each epoch is read
    on its own; cpu_rss_peak_mb is the process's lifetime peak (ru_maxrss).
    """

    def __init__(self, device=None):
        self.device = torch.device(device) if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self._cuda = self.device.type == "cuda"
        self.start()

    def _sync(self):
        """Wait for queued CUDA work so the wall clock covers it."""
        if self._cuda:
            torch.cuda.synchronize(self.device)

    def start(self):
        """Begin an epoch: zero the phase timers and reset the GPU peak counters."""
        self._sync()
        if self._cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        self._t0 = time.perf_counter()
        self._phase = {"train": 0.0, "eval": 0.0}
        self._open = None

    @contextmanager
    def phase(self, name):
        """Time the enclosed block as part of this epoch's train or eval time."""
        self._sync()
        self._open = (name, time.perf_counter())
        try:
            yield
        finally:
            self._sync()
            self._phase[name] += time.perf_counter() - self._open[1]
            self._open = None

    def row(self, n_train_examples):
        """The epoch's RESOURCE_FIELDS so far, including any phase still open."""
        self._sync()
        now = time.perf_counter()
        phase = dict(self._phase)
        if self._open is not None:
            phase[self._open[0]] += now - self._open[1]
        nan = float("nan")
        return {
            "epoch_time_s": now - self._t0,
            "train_time_s": phase["train"],
            "eval_time_s": phase["eval"],
            "s_per_train_example": phase["train"] / n_train_examples if n_train_examples else nan,
            "gpu_mem_peak_mb": torch.cuda.max_memory_allocated(self.device) / 2 ** 20 if self._cuda else nan,
            "gpu_mem_reserved_peak_mb": torch.cuda.max_memory_reserved(self.device) / 2 ** 20 if self._cuda else nan,
            "cpu_rss_peak_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        }
