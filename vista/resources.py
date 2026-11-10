"""
vista.resources
===============
What the machine offers, and how the stacking shares it.

The stacking runs readers (processes), the compute kernel (OpenMP threads,
or the GPU) and writers (threads) at the same time.  Unless the user fixes
them, their numbers come from the cores this process may use, so that a
run takes the whole machine (or the whole allocation of a batch job) and
nothing more:

* readers  = min(entries, max(1, cores / 4), 8)
* writers  = max(1, cores / 4), at most 8 (a Measurement Set: 1)
* threads  = the cores left, at least 1

A number fixed by the user is taken out of the cores first and the others
share what is left; numbers that alone exceed the cores are refused (one
reader, one writer and one thread are always allowed: below three cores the
run needs them anyway).  Oversubscribing does not pay: on two cores one of
each stacked faster than two readers, two writers and two threads.  The
reader processes pass the data through shared memory (``/dev/shm``): when
it is too small for the chunks in flight the readers run as threads.  On a
GPU, ``gpu_batch`` (chunks per launch) is the largest that fits both in the
free memory of the GPU and in a quarter of the free host memory.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional


def available_cores() -> int:
    """Cores this process may use: its CPU affinity, a SLURM allocation
    (``SLURM_CPUS_PER_TASK``) and a cgroup CPU quota (containers), whichever
    is smallest."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit() and int(slurm) > 0:
        n = min(n, int(slurm))
    try:
        with open("/sys/fs/cgroup/cpu.max") as fh:          # cgroup v2: "quota period"
            quota, period = fh.read().split()[:2]
        if quota != "max":
            n = min(n, max(1, math.ceil(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return max(1, n)


def available_memory() -> Optional[int]:
    """Free host memory in bytes (``MemAvailable``, bounded by a cgroup
    limit), or None when it cannot be read."""
    free = None
    try:
        with open("/proc/meminfo") as fh:
            for row in fh:
                if row.startswith("MemAvailable:"):
                    free = int(row.split()[1]) * 1024
                    break
    except (OSError, ValueError):
        return None
    try:
        with open("/sys/fs/cgroup/memory.max") as fh:
            limit = fh.read().strip()
        with open("/sys/fs/cgroup/memory.current") as fh:
            used = int(fh.read().strip())
        if limit != "max" and free is not None:
            free = min(free, max(0, int(limit) - used))
    except (OSError, ValueError):
        pass
    return free


def shared_memory_free() -> Optional[int]:
    """Free space of ``/dev/shm`` in bytes, or None."""
    try:
        st = os.statvfs("/dev/shm")
    except (OSError, AttributeError):
        return None
    return st.f_bavail * st.f_frsize


def _count(name, value):
    if value is None:
        return None
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a whole number, not {value!r}") from None
    if v < 1 or v != value:
        raise ValueError(f"{name} must be a whole number >= 1, not {value!r}")
    return v


@dataclass
class Plan:
    """How a stacking run shares the machine."""

    cores: int
    n_readers: int
    n_writers: int
    n_threads: int
    chosen_by_user: tuple

    def describe(self) -> str:
        mark = lambda k: " (set)" if k in self.chosen_by_user else ""
        return (f"{self.cores} core{'s' if self.cores > 1 else ''}: "
                f"{self.n_readers} reader{'s' if self.n_readers > 1 else ''}{mark('n_readers')}, "
                f"{self.n_threads} kernel thread{'s' if self.n_threads > 1 else ''}"
                f"{mark('n_threads')}, {self.n_writers} writer"
                f"{'s' if self.n_writers > 1 else ''}{mark('n_writers')}")


def plan_workers(n_entries: int, use_zarr: bool, n_readers=None, n_writers=None,
                 n_threads=None, cores: Optional[int] = None) -> Plan:
    """Readers, writers and kernel threads for a run (see the module docstring).

    Raises ValueError when the numbers fixed by the user exceed the cores, or
    when more than one writer is asked for a Measurement Set."""
    n_readers = _count("n_readers", n_readers)
    n_writers = _count("n_writers", n_writers)
    n_threads = _count("n_threads", n_threads)
    cores = available_cores() if cores is None else int(cores)
    given = {k: v for k, v in (("n_readers", n_readers), ("n_writers", n_writers),
                               ("n_threads", n_threads)) if v is not None}
    if not use_zarr and n_writers not in (None, 1):
        raise ValueError(f"n_writers={n_writers}: a Measurement Set is written by one "
                         f"thread (a casacore table cannot be written in parallel); "
                         f"use output_format='zarr' for parallel writers")
    total = sum(given.values())
    if total > max(cores, len(given)):          # one of each is always allowed
        what = " + ".join(f"{k}={v}" for k, v in given.items())
        raise ValueError(
            f"{what}{f' = {total}' if len(given) > 1 else ''}, but this process may use "
            f"{cores} core{'s' if cores > 1 else ''}: lower "
            f"{'them' if len(given) > 1 else 'it'}, or leave it unset (None) and the "
            f"numbers come from the cores")
    left = cores - total
    quarter = max(1, round(cores / 4))
    if n_readers is None:
        n_readers = max(1, min(max(1, n_entries), quarter, 8, max(1, left - 1)))
        left -= n_readers
    if n_writers is None:
        n_writers = max(1, min(quarter, 8, max(1, left - 1))) if use_zarr else 1
        left -= n_writers
    if n_threads is None:
        n_threads = max(1, left)
    return Plan(cores, n_readers, n_writers, n_threads, tuple(given))


def chunk_bytes(rows: int, nchan: int, ncorr: int) -> int:
    """Bytes of one chunk of visibilities and flags (complex64 + bool) and uvw."""
    return int(rows) * (int(nchan) * int(ncorr) * 9 + 24)


def plan_gpu_batch(gpu_batch, bytes_in: int, bytes_out: int, gpu_free: Optional[int],
                   host_free: Optional[int], n_writers: int) -> int:
    """Chunks per GPU launch: the largest that fits in 80% of the free GPU
    memory (input and output of every chunk) and in a quarter of the free
    host memory (the chunks waiting for the launch and the page-locked
    output buffers), at most 32.  A value fixed by the user above that is
    refused."""
    per_chunk = max(1, bytes_in + bytes_out)
    limits = [32]
    if gpu_free:
        limits.append(int(0.8 * gpu_free // per_chunk))
    if host_free:
        # waiting chunks (>= batch) + output buffers (batch + writers + 4)
        limits.append(int((0.25 * host_free - (n_writers + 4) * bytes_out) // per_chunk))
    best = max(1, min(limits))
    user = _count("gpu_batch", gpu_batch)
    if user is None:
        return best
    if user > best:
        raise ValueError(
            f"gpu_batch={user}: at most {best} chunks of {per_chunk / 2**20:.0f} MB fit per "
            f"launch here ({'GPU' if gpu_free and int(0.8 * gpu_free // per_chunk) == best else 'host'} "
            f"memory); leave it unset to use that")
    return user


def reader_mode(n_readers: int, queue_chunks: int, bytes_in: int) -> str:
    """'process' (parallel reads through shared memory) when ``/dev/shm``
    holds the chunks in flight with a margin, otherwise 'thread'."""
    free = shared_memory_free()
    need = 2 * (queue_chunks + n_readers) * bytes_in
    if free is not None and free < need:
        return "thread"
    return "process"


__all__ = ["available_cores", "available_memory", "shared_memory_free", "Plan",
           "plan_workers", "plan_gpu_batch", "chunk_bytes", "reader_mode"]
