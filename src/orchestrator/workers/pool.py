"""GPU worker pool — subprocess spawn is blocked until Phase 5.

Does not run CV. When implemented, workers will call `python -m viana` only.
"""

from __future__ import annotations

MAX_CONCURRENT_GPU_JOBS = 2
GPU_DEVICES: tuple[str, str] = ("cuda:0", "cuda:1")


class WorkerPool:
    """Assigns at most two GPUs. Does not spawn engine processes yet."""

    def assign_gpu(self, occupied: set[str]) -> str | None:
        """Return the next free `cuda:0` or `cuda:1`, or None if both busy."""
        for device in GPU_DEVICES:
            if device not in occupied:
                return device
        return None

    def spawn_run(self, job_id: str, gpu_device: str) -> None:
        """Spawn `python -m viana run` — not implemented until Phase 5."""
        raise NotImplementedError(
            f"spawn python -m viana run blocked until Phase 5 ({job_id} on {gpu_device})"
        )
