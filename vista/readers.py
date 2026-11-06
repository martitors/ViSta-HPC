"""
vista.readers -- the readers of the stacking
============================================

Each reader takes the next entry from a task queue and reads it with
:func:`vista.ms_input.read_entry_chunks`, sending what it reads to the main
process through one bounded queue.  Readers are separate processes by
default: python-casacore holds the Python GIL while reading, so reader
threads would read one at a time.  The visibilities and flags of a chunk
travel through shared memory (``/dev/shm``), without copies.

    ReaderPool       the readers of one run (processes or threads)
    reader_process   entry point of a reader process
    reader_thread    the same, as a thread (reader_mode="thread")

Messages to the main process: ``(kind, reader_id, index, obj, seconds)``

    "info"   once per entry, before its chunks: obj = EntryInfo
    "chunk"  obj = Chunk; from a process, its vis and flag are
             (shared-memory name, shape, dtype): the main process maps them,
             uses them and removes them (ReaderPool.arrays / release)
    "skip"   the entry gives no usable data: obj = the reason
    "error"  unexpected exception in the reader: obj = the traceback
    "done"   the reader has no more entries

Internal module of :mod:`vista.pipeline`.
"""

import contextlib
import dataclasses
import queue
import sys
import threading
import traceback

import numpy as np

from .ms_input import read_entry_chunks

_POLL_S = 0.2          # how often a blocked reader checks the stop event [s]


# =============================================================================
# Shared memory
# =============================================================================

def _shm_create(nbytes):
    """New shared-memory segment, not tracked by the resource tracker.

    The main process removes it once the chunk is used, so the reader must
    not remove it when it exits.
    """
    from multiprocessing import shared_memory
    try:
        # Python >= 3.13: no resource tracker, the main process frees the segment
        return shared_memory.SharedMemory(create=True, size=max(1, nbytes), track=False)
    except TypeError:
        shm = shared_memory.SharedMemory(create=True, size=max(1, nbytes))
        try:
            from multiprocessing import resource_tracker
            resource_tracker.unregister(shm._name, "shared_memory")
        except Exception:
            pass
        return shm


def _shm_attach(name):
    """Map a segment created by a reader.

    (Python < 3.13: attaching registers the segment in the resource tracker,
    and unlink() unregisters it.)
    """
    from multiprocessing import shared_memory
    try:
        return shared_memory.SharedMemory(name=name, track=False)
    except TypeError:
        return shared_memory.SharedMemory(name=name)


def _shm_unlink_untracked(shm):
    """Remove a segment that is not in the resource tracker (created by a reader)."""
    try:
        import _posixshmem
        _posixshmem.shm_unlink(shm._name)
    except ImportError:
        shm.unlink()


def release(segments):
    """Unmap and remove the shared-memory segments of a used chunk."""
    for shm in segments:
        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
        except Exception:
            pass


# =============================================================================
# The readers
# =============================================================================

def _put(out_q, msg, stop_ev):
    """Put *msg* on *out_q*; False if the run was stopped before it could."""
    while not stop_ev.is_set():
        try:
            out_q.put(msg, timeout=_POLL_S)
            return True
        except queue.Full:
            continue
    return False


def _next_job(task_q, stop_ev):
    """Next EntryJob from *task_q*; None at the end or when the run is stopped."""
    while not stop_ev.is_set():
        try:
            return task_q.get(timeout=_POLL_S)
        except queue.Empty:
            continue
    return None


def reader_process(reader_id, task_q, out_q, stop_ev):
    """Entry point of a reader process (module level, for 'spawn').

    Reads the entries of *task_q* until a ``None`` arrives, the chunks into
    new shared-memory segments; a chunk message carries the names of its
    segments instead of the arrays.
    """
    import casacore.tables       # noqa: F401  (heavy imports while the parent does the setup)
    import astropy.coordinates   # noqa: F401

    fresh = []          # segments created for the current chunk
    pending = []        # segments already sent, local mapping still to close

    def alloc(shape, dtype):
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        shm = _shm_create(nbytes)
        fresh.append(shm)
        return np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    def try_close():
        still = []
        for shm in pending:
            try:
                shm.close()
            except BufferError:          # the generator still holds a view
                still.append(shm)
        pending[:] = still

    try:
        while True:
            job = _next_job(task_q, stop_ev)
            if job is None:
                break
            try:
                for kind, index, obj, seconds in read_entry_chunks(job, alloc):
                    if kind == "chunk":
                        shm_v, shm_f = fresh[0], fresh[1]
                        obj = dataclasses.replace(
                            obj, vis=(shm_v.name, obj.vis.shape, "complex64"),
                            flag=(shm_f.name, obj.flag.shape, "bool"))
                    sent = _put(out_q, (kind, reader_id, index, obj, seconds), stop_ev)
                    if kind == "chunk":
                        if not sent:              # stopped: nobody will free them
                            for shm in fresh:
                                try:
                                    _shm_unlink_untracked(shm)
                                except Exception:
                                    pass
                        pending.extend(fresh)
                        fresh.clear()
                    try_close()
                    if not sent:
                        break
            except RuntimeError as e:
                _put(out_q, ("skip", reader_id, job.index, str(e), 0.0), stop_ev)
    except Exception:
        _put(out_q, ("error", reader_id, None, traceback.format_exc(), 0.0), stop_ev)
    finally:
        for shm in fresh:                 # created but never sent: removed here
            try:
                _shm_unlink_untracked(shm)
            except Exception:
                pass
            try:
                shm.close()
            except Exception:
                pass
        try_close()
        _put(out_q, ("done", reader_id, None, None, 0.0), stop_ev)


def reader_thread(reader_id, task_q, out_q, stop_ev):
    """A reader as a thread (reader_mode="thread"): plain NumPy arrays, same messages."""
    alloc = lambda shape, dtype: np.empty(shape, dtype)
    try:
        while True:
            job = _next_job(task_q, stop_ev)
            if job is None:
                break
            try:
                for kind, index, obj, seconds in read_entry_chunks(job, alloc):
                    _put(out_q, (kind, reader_id, index, obj, seconds), stop_ev)
            except RuntimeError as e:
                _put(out_q, ("skip", reader_id, job.index, str(e), 0.0), stop_ev)
    except Exception:
        _put(out_q, ("error", reader_id, None, traceback.format_exc(), 0.0), stop_ev)
    finally:
        _put(out_q, ("done", reader_id, None, None, 0.0), stop_ev)


@contextlib.contextmanager
def _spawn_without_main():
    """Start 'spawn' processes without re-running the user's script in them.

    A spawned child normally imports the parent's __main__ module again: a
    script that calls ViSta.run() without an ``if __name__ == "__main__":``
    guard would start the whole run in every reader.  The readers need only
    this module, so __main__ is hidden while they start.
    """
    main = sys.modules.get("__main__")
    saved = {}
    if main is not None:
        for attr in ("__file__", "__spec__"):
            if hasattr(main, attr):
                saved[attr] = getattr(main, attr)
        if "__file__" in saved:
            del main.__file__
        if "__spec__" in saved:
            main.__spec__ = None
    try:
        yield
    finally:
        for attr, value in saved.items():
            setattr(main, attr, value)


class ReaderPool:
    """The readers of one run, started at once and fed with EntryJobs.

    Parameters
    ----------
    mode : {'process', 'thread'}
        ``'process'``: parallel reads, chunks through shared memory.
        ``'thread'``: no shared memory (e.g. a container with a small
        ``/dev/shm``), but the reads take turns.
    n_readers : int
        Readers, each reading a different entry.
    queue_size : int
        Messages that may wait for the main process; bounds the memory used
        by chunks already read.

    Attributes
    ----------
    messages : queue
        Where the readers put their messages (see the module docstring).
    """

    def __init__(self, mode, n_readers, queue_size):
        self.mode = mode
        self.n_readers = n_readers
        self._workers = []
        if mode == "process":
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            self._tasks = ctx.Queue()
            self.messages = ctx.Queue(maxsize=queue_size)
            self._stop = ctx.Event()
            self._workers = [ctx.Process(target=reader_process,
                                         args=(k, self._tasks, self.messages, self._stop),
                                         name=f"vista-reader-{k}", daemon=True)
                             for k in range(n_readers)]
            # started now, so that they import numpy / casacore while the
            # main process does the setup; their jobs arrive later
            with _spawn_without_main():
                for p in self._workers:
                    p.start()
        else:
            self._tasks = queue.Queue()
            self.messages = queue.Queue(maxsize=queue_size)
            self._stop = threading.Event()
            self._workers = [threading.Thread(target=reader_thread,
                                              args=(k, self._tasks, self.messages, self._stop),
                                              name=f"vista-reader-{k}", daemon=True)
                             for k in range(n_readers)]
            for t in self._workers:
                t.start()

    def submit(self, jobs):
        """Queue the entries to read, then one end marker per reader."""
        for job in jobs:
            self._tasks.put(job)
        for _ in range(self.n_readers):
            self._tasks.put(None)

    def arrays(self, chunk):
        """VIS and FLAG of a chunk, and the shared-memory segments to release.

        Returns
        -------
        (ndarray, ndarray, list)
            The arrays (views on shared memory from a process) and the
            segments, to pass to :func:`release` once the arrays are no longer
            used (empty for a thread reader).
        """
        if self.mode == "thread":
            return chunk.vis, chunk.flag, []
        segments, arrays = [], []
        for name, shape, dtype in (chunk.vis, chunk.flag):
            shm = _shm_attach(name)
            segments.append(shm)
            arrays.append(np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf))
        return arrays[0], arrays[1], segments

    def stop(self):
        """Stop the readers (at the end of the run, or after an error).

        Once they have exited, the chunks still in the queue (only after an
        error) are freed.
        """
        self._stop.set()
        for w in self._workers:
            w.join(timeout=5)
            if self.mode == "process" and w.is_alive():
                w.terminate()
        self.discard_queued()

    def discard_queued(self):
        """Stop the readers and free the chunks still queued (after an error)."""
        self._stop.set()
        while True:
            try:
                msg = self.messages.get_nowait()
            except Exception:
                break
            if self.mode == "process" and msg and msg[0] == "chunk":
                for name, _, _ in (msg[3].vis, msg[3].flag):
                    try:
                        release([_shm_attach(name)])
                    except Exception:
                        pass
