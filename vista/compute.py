"""
vista.compute -- the calls to the compiled kernel
=================================================

The kernel ``ms_ops`` (C++/OpenMP, optionally CUDA; src/ms_ops.cpp and
src/ms_ops_cuda.cu) does the three steps of the stacking on a chunk of rows:
rest-framing, re-centring and rebinning.  It releases the Python GIL, so the
readers and the writer keep working while it runs.

    run_cpu        one chunk on the CPU
    GpuEngine      many chunks per GPU launch, into reused page-locked buffers
    kernel_inputs  redshift and frequencies for the kernel, with or without
                   the frame correction (vista.frames)
    set_threads, cpu_threads, check_gpu, gpu_memory_gb
                   settings and checks, for ViSta.run()

This is the only module that imports ``ms_ops``.

Internal module of :mod:`vista.pipeline`.
"""

from queue import Queue

import numpy as np

from .common import detail, warn

try:
    from . import ms_ops            # installed into vista/ by 'make install'
except ImportError:                 # or found on PYTHONPATH
    import ms_ops


# =============================================================================
# Settings and checks
# =============================================================================

def set_threads(n):
    """Use *n* OpenMP threads in the CPU kernel."""
    ms_ops.omp_set_threads(int(n))


def cpu_threads():
    """OpenMP threads the CPU kernel uses."""
    return ms_ops.omp_max_threads()


def check_gpu():
    """Raise RuntimeError, with the reason, if the kernel cannot run on a GPU."""
    if not ms_ops.built_with_cuda():
        raise RuntimeError("device='gpu': the ms_ops kernel was built without "
                           "CUDA; rebuild it with cmake -DWITH_CUDA=ON")
    if not ms_ops.gpu_available():
        raise RuntimeError("device='gpu': no GPU visible to CUDA "
                           "(check CUDA_VISIBLE_DEVICES and the node)")


def gpu_memory_gb():
    """Total memory of the GPU [GiB]."""
    _free, total = ms_ops.gpu_memory_info()
    return total / 2**30


# =============================================================================
# CPU
# =============================================================================

def kernel_inputs(job, info, k=None):
    """Redshift, input frequencies and widths passed to the kernel.

    Without a frame correction: z and the rest-frame input channels.  With a
    Doppler factor k: the redshift (1+z)*k - 1 and the channels times k.  For
    the kernel this is a source at a slightly different redshift, which is
    exactly what the frame correction is: the phase shift keeps using the
    observed frequency (the factors cancel), the UVW and the rebinning are
    those of the chosen frame.
    """
    if k is None:
        return job.z, info.freq, info.width
    return (1.0 + job.z) * k - 1.0, info.freq * k, info.width * k


def run_cpu(vis, flag, uvw, job, info, segments=None):
    """Rest-frame, re-centre and rebin one chunk on the CPU.

    Parameters
    ----------
    vis, flag : ndarray
        Visibilities and flags of the chunk, shape (rows, channels, corr).
    uvw : ndarray
        UVW of the rows [m].
    job : EntryJob
        The entry (redshift, phase centre, output channels).
    info : EntryInfo
        What the reader found (source position, input channels).
    segments : list of (int, int, float) or None
        With a frame correction, the runs of rows with the same Doppler
        factor (see vista.frames.row_segments); None: one call for the whole
        chunk.  The rows with the same k go in one kernel call: one run in
        an MS in time order, several when the rows are not in time order
        (e.g. sorted by baseline), which are then gathered.

    Returns
    -------
    (ndarray, ndarray, ndarray)
        Visibilities and flags on the output channels, and the rest-frame
        UVW, rotated to the source.
    """
    ra0, dec0 = job.meta.phase_dir

    def kernel(v, f, u, k):
        z, freq, width = kernel_inputs(job, info, k)
        return ms_ops.full_pipeline(v, f, u, z, freq, ra0, dec0, info.ra1, info.dec1,
                                    job.freq_new, width_old_rf=width)

    if segments is None:
        return kernel(vis, flag, uvw, None)
    if len(segments) == 1:
        return kernel(vis, flag, uvw, segments[0][2])
    runs_of = {}                                # k -> its runs of rows
    for a, b, k in segments:
        runs_of.setdefault(k, []).append((a, b))
    out = None
    for k, runs in runs_of.items():
        if len(runs) == 1:
            rows = slice(*runs[0])
            part = kernel(vis[rows], flag[rows], uvw[rows], k)
        else:
            rows = np.concatenate([np.arange(a, b) for a, b in runs])
            part = kernel(vis[rows], flag[rows], uvw[rows], k)
        if out is None:
            n = vis.shape[0]
            out = tuple(np.empty((n,) + p.shape[1:], p.dtype) for p in part)
        for o, p in zip(out, part):
            o[rows] = p
    return out


# =============================================================================
# GPU
# =============================================================================

class _OutPool:
    """Output buffers of one shape, reused for every chunk.

    A fixed set of slots (vis, flag, uvw) as large as a chunk, allocated once
    and page-locked: the GPU copies the results straight into them (DMA) and
    no new memory pages are created for every chunk.  The compute takes a
    free slot, the writer gives it back once written; with no free slot the
    compute waits for the writer.
    """

    def __init__(self, nslots, rows, nchan, ncorr, pin=True):
        self.rows = rows
        self.slots = []
        self.free = Queue()
        self.pinned = 0
        for k in range(nslots):
            sl = (np.zeros((rows, nchan, ncorr), np.complex64),   # zeros: pages touched
                  np.zeros((rows, nchan, ncorr), np.bool_),
                  np.zeros((rows, 3), np.float64))
            if pin:
                self.pinned += int(all(ms_ops.pin_host(a) for a in sl))
            self.slots.append(sl)
            self.free.put(k)

    def views(self, k, n):
        v, f, u = self.slots[k]
        return v[:n], f[:n], u[:n]

    def close(self):
        if self.pinned:
            for sl in self.slots:
                for a in sl:
                    try:
                        ms_ops.unpin_host(a)
                    except Exception:
                        pass
        self.slots = []


class GpuEngine:
    """Many chunks per GPU launch, into reused page-locked output buffers.

    Parameters
    ----------
    nslots : int
        Output buffers per number of correlations: enough for the chunks
        waiting for the writer and those of one launch.
    rows : int
        Rows of a buffer (``chunk_rows``); a larger chunk gets new arrays.
    nchan : int
        Output channels.

    Attributes
    ----------
    launches, chunks : int
        GPU launches done, chunks computed.
    failed : bool
        True once the GPU has failed (the kernel then continues on the CPU).
    """

    def __init__(self, nslots, rows, nchan):
        self.nslots, self.rows, self.nchan = int(nslots), int(rows), int(nchan)
        self.pools = {}                 # ncorr -> _OutPool
        self.launches = 0
        self.chunks = 0
        self.failed = False

    def take(self, nrows, ncorr, wait):
        """A free output buffer for a chunk.

        Parameters
        ----------
        nrows, ncorr : int
            Shape of the chunk.
        wait : callable
            ``wait(queue)`` returns the next item of the queue, waiting for
            the writer to give a buffer back if none is free.

        Returns
        -------
        (int, int) or None
            The buffer (ncorr, slot), or None when the chunk is larger than
            the buffers.
        """
        pool = self.pools.get(int(ncorr))
        if pool is None:
            pool = _OutPool(self.nslots, self.rows, self.nchan, int(ncorr))
            self.pools[int(ncorr)] = pool
            detail(f"GPU: {self.nslots} output buffers of {self.rows} rows "
                   f"({pool.pinned}/{self.nslots} page-locked)")
        if nrows > pool.rows:
            return None
        return (int(ncorr), wait(pool.free))

    def give_back(self, slot):
        """Make a buffer free again (after the writer has written it)."""
        if slot is not None:
            self.pools[slot[0]].free.put(slot[1])

    def outputs(self, slot, nrows, ncorr):
        """(vis, flag, uvw) output arrays: views on the buffer, or new arrays."""
        if slot is None:
            return (np.empty((nrows, self.nchan, ncorr), np.complex64),
                    np.empty((nrows, self.nchan, ncorr), np.bool_),
                    np.empty((nrows, 3), np.float64))
        return self.pools[slot[0]].views(slot[1], nrows)

    def run(self, vis, flag, uvw, jobs, infos, outs, factors=None):
        """One GPU launch for several blocks of rows (possibly of different entries).

        Parameters
        ----------
        vis, flag, uvw : list of ndarray
            The input of each block.
        jobs, infos : list
            The EntryJob and EntryInfo of each block.
        outs : list of (ndarray, ndarray, ndarray)
            Where the results go (from :meth:`outputs`).
        factors : list of float or None, or None
            Doppler factor of each block (frame correction), None for none.
        """
        phase = [j.meta.phase_dir for j in jobs]
        factors = factors or [None] * len(jobs)
        inputs = [kernel_inputs(j, i, k) for j, i, k in zip(jobs, infos, factors)]
        ms_ops.full_pipeline_batch_into(
            vis, flag, uvw, [x[0] for x in inputs], [x[1] for x in inputs],
            [p[0] for p in phase], [p[1] for p in phase],
            [i.ra1 for i in infos], [i.dec1 for i in infos],
            [j.freq_new for j in jobs],
            [o[0] for o in outs], [o[1] for o in outs], [o[2] for o in outs],
            width_old_list=[x[2] for x in inputs])
        self.launches += 1
        self.chunks += len(jobs)
        if not self.failed and not ms_ops.gpu_available():
            self.failed = True
            warn(f"GPU error, the run continues on the CPU: {ms_ops.gpu_last_error()}")

    def close(self):
        """Unlock and drop the buffers."""
        for pool in self.pools.values():
            pool.close()
        self.pools.clear()
