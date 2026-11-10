"""
vista.pipeline -- visibility-domain stacking
=============================================

ViSta combines interferometric Measurement Sets (MSs) of sources at different
redshifts directly in the visibility (uv) plane.  Every input dataset goes
through three steps, all done by the compiled kernel ``ms_ops``:

    1. Rest-framing   the baselines (UVW) are divided by (1+z) and the
                      frequencies multiplied by (1+z), so every source is seen
                      in its own rest frame.
    2. Re-centring    the visibilities are phase-shifted so that the source
                      sits at the phase centre.
    3. Rebinning      the channels are averaged onto a common rest-frame grid
                      with overlap weights, as CASA ``mstransform`` does.

The result is one stack with one spectral window per line of the input list:
a Zarr store, written in parallel (``vista.export_ms`` turns it into a
Measurement Set for tclean or WSClean), or directly a Measurement Set (an
output path ending in ``.ms``, or ``output_format="ms"``).  ``vista.extract``
reads both.


How a run is organised
----------------------
::

    input list ──> metadata (SPECTRAL_WINDOW, FIELD, DATA_DESCRIPTION)
                       │
                       ├──> common spectral grid, one window per entry
                       └──> output subtables (ANTENNA, FIELD, ...)

    readers (N processes)  read the entries chunk by chunk, only the     vista.readers
         │ queue           rows and channels needed; VIS / FLAG travel
         │                 through shared memory
    main thread            kernel: rest-frame, shift, rebinning; on the   _compute_loop
         │ queue           GPU all the chunks already waiting (up to
         │                 gpu_batch) go in one launch
    writers                Zarr: n_writers threads, each chunk at its     _writer_loop
                           place in the store; MS: one thread appends
                           the rows to the table

    then: subtables, optional move from scratch, summary

The readers are processes because python-casacore holds the Python GIL while
reading: threads would read one at a time.  The kernel releases the GIL, so
reading and writing never stop while it runs.

A casacore table can be written by one thread only, and with an MS output
that single writer usually sets the pace of the whole run.  In a Zarr store
every chunk of every column is a file of its own, so several threads write
at once, and the columns are compressed (the empty, flagged channels of a
partly covered grid take almost no space).  In the store the rows of every
entry are in input order; in an MS they are in the order the chunks are
processed (the content does not depend on it).


Where the code is
-----------------
    pipeline.py        the Stacker: input list, the run, its threads
    resources.py       cores, memory, readers / writers / threads, GPU batch
    spectral_grid.py   the common rest-frame channel grid
    ms_input.py        reading the input MSs (metadata, rows, SPW merging)
    frames.py          spectral frame of the inputs, Doppler factor per row
    polarization.py    the correlations of the inputs
    readers.py         the reader processes, shared memory
    compute.py         the calls to the kernel, GPU buffers
    ms_output.py       writing the output MS (main table, subtables)
    stackio.py         the Zarr store (writer, reader of the extraction),
                       export_ms
    common.py          log, lock, coordinates, small helpers
    ms_ops             the compiled kernel (src/ms_ops.cpp, src/ms_ops_cuda.cu)
"""

# Threads of OpenMP / OpenBLAS / MKL.  They are read once, when the libraries
# are loaded, so they are set here, before NumPy and the kernel are imported.
# A value already set by the user (or by SLURM) is kept; the run sets the
# kernel threads afterwards (vista.resources).
import os

_n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or (os.cpu_count() or 4)
os.environ.setdefault("OMP_NUM_THREADS",        str(_n_threads))
os.environ.setdefault("OPENBLAS_NUM_THREADS",   os.environ["OMP_NUM_THREADS"])
os.environ.setdefault("MKL_NUM_THREADS",        os.environ["OMP_NUM_THREADS"])
os.environ.setdefault("NUMEXPR_NUM_THREADS",    os.environ["OMP_NUM_THREADS"])
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", os.environ["OMP_NUM_THREADS"])
# No OpenMP thread binding by default: with OMP_PROC_BIND set, loading the
# kernel pins the main thread to one core, and every reader process and
# writer thread started afterwards inherits that single core.
try:
    _AFFINITY = os.sched_getaffinity(0)
except (AttributeError, OSError):
    _AFFINITY = None

import gc
import logging
import shutil
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import Optional

import numpy as np
import dask

from . import __version__, _logger, _reset_clock
from . import compute, ms_output
from .common import detail, fmt_size, fmt_time, log, warn
from .ms_input import EntryJob, read_ms_metadata
from .readers import ReaderPool, release
from .spectral_grid import C_LIGHT_KMS, compute_grid
from .stackio import ZarrStackWriter, store_size

# a binding set by the user (OMP_PROC_BIND) has now pinned the main thread:
# give it back all its cores, for the readers and writers it starts
if _AFFINITY is not None:
    try:
        if os.sched_getaffinity(0) != _AFFINITY:
            os.sched_setaffinity(0, _AFFINITY)
    except OSError:
        pass

_DONE = object()            # end of the stream of processed chunks


class _Abort(Exception):
    """Raised in a thread waiting on a queue when another thread has failed."""


@dataclass
class _Processed:
    """One processed chunk, waiting for a writer."""
    columns: dict                       # DATA, FLAG, UVW, ANTENNA1, ... of the rows
    index: int                          # its entry (line of the input list)
    row0: int                           # its first row among the rows of the entry
    slot: Optional[tuple] = None        # GPU buffer to give back once written
    last: bool = False                  # last chunk of its entry


@dataclass
class _Run:
    """State of one call of ViSta.run(), shared by the main and writer threads."""
    output: str                         # where the stack is written
    jobs: list                          # EntryJob of every line of the input list
    readers: ReaderPool
    out_q: Queue                        # processed chunks waiting for the writers
    store: Optional[ZarrStackWriter] = None     # Zarr output; None: MS output
    n_writers: int = 1
    lock: threading.Lock = field(default_factory=threading.Lock)  # writers' counters
    gpu: Optional[compute.GpuEngine] = None
    gpu_batch: int = 1
    inline: Optional[object] = None     # vista.extract.inline.InlineExtraction
    abort: threading.Event = field(default_factory=threading.Event)
    errors: list = field(default_factory=list)
    infos: dict = field(default_factory=dict)       # index -> EntryInfo
    skipped: dict = field(default_factory=dict)     # index -> reason
    rows: int = 0
    bytes: int = 0
    flagged: int = 0
    values: int = 0
    ms_done: int = 0
    t_read: float = 0.0
    t_compute: float = 0.0
    t_write: float = 0.0
    t_first: Optional[float] = None
    t_loop: float = 0.0
    t_loop_end: float = 0.0

    def put(self, q, item):
        """Put *item* on *q*; raises _Abort if the run fails meanwhile."""
        while True:
            if self.abort.is_set():
                raise _Abort()
            try:
                q.put(item, timeout=0.2)
                return
            except Full:
                continue

    def get(self, q):
        """Next item of *q*; raises _Abort if the run fails meanwhile."""
        while True:
            if self.abort.is_set():
                raise _Abort()
            try:
                return q.get(timeout=0.2)
            except Empty:
                continue

    def fail(self, error):
        """Record *error* and stop the other threads."""
        self.errors.append(error)
        self.abort.set()


class Stacker:
    """Visibility-domain stacking of a list of Measurement Sets (internal:
    the user drives it through :class:`vista.ViSta`).

    Reader processes, the compute kernel and the writers run at the same
    time, connected by bounded queues (see the module docstring).  The
    kernel (``ms_ops``) is written in C++/OpenMP; with a CUDA build it can
    run on a GPU (``run(device="gpu")``).

    Parameters
    ----------
    input_file : str
        Text file with one entry per line::

            <ms_path>  <redshift>  <RA>  <Dec>  [FIELD_ID]  [SPW_IDS]  [NORM]

        RA and Dec are those of the source to stack (sexagesimal or decimal).
        ``FIELD_ID`` selects one field; without it every field is read, with
        the phase centre of field 0.  ``SPW_IDS`` is a comma-separated list;
        without it every SPW is used.  ``NORM`` is only used by
        ``vista.extract``.  Text after ``#`` is a comment.
    verbose : bool or 'debug'
        Passed on to the steps run during the stacking.
    """

    # ---- input list -----------------------------------------------------------

    def __init__(self, input_file, verbose=False, chunk_rows=5000):
        self.input_file  = input_file
        self.chunk_rows  = chunk_rows        # at most; fewer for wide spectra (ms_input)
        self.verbose     = verbose
        self.data_column = "auto"
        self.frame = None
        self.ms_list, self.z_list = [], []
        self.ra_list, self.dec_list = [], []
        self.field_list, self.spw_list, self.norm_list = [], [], []
        self._load_input(input_file)

    def _load_input(self, path):
        """Read the input list into the per-entry lists.

        Parameters
        ----------
        path : str
            The input file (see the class docstring for the format).  The
            optional last value, ``NORM``, is recognised because it is neither
            an integer nor a list of integers, so it cannot be confused with
            FIELD_ID or SPW_IDS; it can also be written as ``norm=2.91e13``.
        """
        from .extract.sources import split_trailing_columns

        with open(path) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 4:
                    raise ValueError(f"Malformed input line: {line!r}")
                field, spws, norm = split_trailing_columns(parts[4:])
                self.ms_list.append(parts[0])
                self.z_list.append(float(parts[1]))
                self.ra_list.append(parts[2])
                self.dec_list.append(parts[3])
                self.field_list.append(field)
                self.spw_list.append(spws)
                self.norm_list.append(norm)

    # ---- metadata and spectral grid ---------------------------------------------

    def _compute_per_ms_grids(self, central_freq, nchan_out=None,
                              velocity_range_kms=None, channel_width_hz=None):
        """Read the metadata of every entry and compute the output grid.

        The metadata (kept in ``self._meta``) is read in parallel; the grid
        (kept in ``self._grid``) is described in
        :func:`vista.spectral_grid.compute_grid`.

        Parameters
        ----------
        central_freq : float
            Rest-frame centre of the grid [Hz].
        nchan_out : int or None, default None
            Number of channels; ``None`` = the widest coverage of one entry.
        velocity_range_kms : float, (float, float) or None, default None
            Bandwidth as a velocity range; overrides *nchan_out*.
        channel_width_hz : float or None, default None
            Channel width; ``None`` = the widest channel of the sample.

        Returns
        -------
        (list of ndarray, float)
            The channel centres of each entry [Hz] and the channel width [Hz].
        """
        from concurrent.futures import ThreadPoolExecutor

        n_ms = len(self.ms_list)
        names = [os.path.basename(m) for m in self.ms_list]
        detail(f"Reading metadata from {n_ms} MSs (SPECTRAL_WINDOW / FIELD / DATA_DESCRIPTION)...")

        def read(i):
            meta = read_ms_metadata(self.ms_list[i], self.field_list[i], self.spw_list[i])
            detail(f"  [{i+1}/{n_ms}]  {names[i]}  spw={meta.spws}  "
                   f"{sum(len(f) for f in meta.freq)} channels")
            return meta

        with ThreadPoolExecutor(max_workers=min(4, n_ms)) as pool:
            self._meta = list(pool.map(read, range(n_ms)))
        self._check_frames()

        df_rest, nchan_in, ranges = [], [], []
        for i, (z, meta) in enumerate(zip(self.z_list, self._meta)):
            freq = np.concatenate(meta.freq) if meta.freq else np.array([0.0])
            width = np.concatenate(meta.width) if meta.width else np.array([1.0])
            nchan_in.append(len(freq))
            if self._doppler[i] is None:
                df_rest.append(float(abs(width[0])) * (1.0 + z))
                freq_rf = freq * (1.0 + z)
            else:               # frequencies in the chosen frame, at mid-observation
                k_mid = self._doppler[i].k_mid
                df_rest.append(float(abs(width[0])) * (1.0 + z) * k_mid)
                freq_rf = freq * (1.0 + z) * k_mid
            ranges.append((float(freq_rf.min()), float(freq_rf.max())))
            detail(f"  [{i+1}/{n_ms}]  {names[i]}  z={z:.5f}  nchan={len(freq)}  "
                   f"df_rf={df_rest[-1]/1e3:.3f} kHz  "
                   f"range=[{freq_rf.min()/1e6:.3f}, {freq_rf.max()/1e6:.3f}] MHz")

        self._grid = compute_grid(central_freq, ranges, df_rest, nchan_in, names,
                                  self.z_list, nchan_out=nchan_out,
                                  velocity_range_kms=velocity_range_kms,
                                  channel_width_hz=channel_width_hz)
        return self._grid.freq, self._grid.df

    def _check_frames(self):
        """Correlations and spectral frame of every entry, and its Doppler
        factor if needed.

        Sets ``self._doppler``: for every entry None (its SPWs are already in
        ``self.frame``) or a :class:`vista.frames.EntryFrame` (they are in
        another frame).  Every k is computed here, in the main process and in
        a fixed order, so that a run is reproducible bit for bit.
        """
        from .common import casacore_lock, parse_ra_dec
        from .frames import check_frames
        from .polarization import check_correlations

        # the same number of correlations in every entry (else the writing fails)
        for level, text in check_correlations(self.ms_list, self.spw_list, lock=casacore_lock):
            {"log": log, "warn": warn, "detail": detail}[level](text)
        coords = [parse_ra_dec(r, d) for r, d in zip(self.ra_list, self.dec_list)]
        self._doppler, messages = check_frames(
            self.ms_list, self.spw_list, [c[0] for c in coords], [c[1] for c in coords],
            self.frame, lock=casacore_lock)
        for level, text in messages:
            {"log": log, "warn": warn, "detail": detail}[level](text)

    def _make_jobs(self, ant_offsets):
        """One :class:`~vista.ms_input.EntryJob` per line of the input list."""
        n_ms = len(self.ms_list)
        jobs = []
        for i in range(n_ms):
            jobs.append(EntryJob(
                index=i, n_entries=n_ms, ms=self.ms_list[i], z=float(self.z_list[i]),
                ra=self.ra_list[i], dec=self.dec_list[i],
                field=self.field_list[i], spws=self.spw_list[i],
                meta=self._meta[i], freq_new=self._grid.freq[i],
                df_new=float(self._grid.df), chunk_rows=int(self.chunk_rows),
                ant_offset=int(ant_offsets[i]), data_column=self.data_column,
                frame=self._doppler[i]))
        return jobs

    # ---- the run -----------------------------------------------------------------

    def run(self, output, central_freq, scratch_dir=None, nchan_out=None,
            velocity_range_kms=None, channel_width_hz=None,
            data_column="auto", frame="LSRK", device="cpu", n_readers=None, n_threads=None,
            gpu_batch=None, output_format=None, n_writers=None, line=None, continuum=None,
            stats_line=None, stats_continuum=None, binning=None, max_amplitude=None):
        """Stack all the entries and write the stack.

        Parameters
        ----------
        output : str
            Where the stack is written: a Zarr store or a Measurement Set.
            Anything already at this path is deleted.
        central_freq : float
            Rest-frame centre of the output grid [Hz], normally the rest
            frequency of the line.
        scratch_dir : str or None
            Write the stack here first (a fast local disk) and move it to
            *output* at the end.
        nchan_out : int or None
            Number of output channels.  ``None`` takes the largest number any
            single entry can fill.
        velocity_range_kms : float, (float, float) or None
            Output bandwidth as a velocity range around *central_freq*
            (``2000``: -2000 to +2000 km/s).  Overrides *nchan_out*.
        channel_width_hz : float or None
            Rest-frame channel width.  ``None`` takes the widest channel of
            the sample, the finest grid every dataset supports.
        data_column : str
            Column to stack: ``'auto'`` takes CORRECTED_DATA when present,
            otherwise DATA.
        frame : str
            The frame of the redshifts (``'LSRK'``, ``'BARY'``, ...): the
            input frequencies are always brought to it, row by row.
        device : {'cpu', 'gpu'}
            Where the kernel runs (``'gpu'`` needs a CUDA build and a GPU).
        n_readers, n_writers, n_threads : int or None
            Reader processes, writer threads (Zarr) and kernel threads.
            ``None``: from the cores available (:mod:`vista.resources`).
        gpu_batch : int or None
            GPU only: chunks per launch.  ``None``: as many as fit in the
            memory of the GPU and of the host.
        output_format : {'zarr', 'ms'} or None
            ``None``: ``'ms'`` when *output* ends in ``.ms``, otherwise
            ``'zarr'``.
        line, continuum, stats_line, stats_continuum, binning, max_amplitude
            Continuum subtraction and compression while the stack is written
            (:mod:`vista.extract.inline`): the lines
            (:class:`~vista.extract.config.Lines`), the subtraction
            (``ContinuumConfig``, None: none), the statistics files and the
            annuli.

        Returns
        -------
        dict
            ``output``, ``format``, ``n_rows``, ``n_spw``, ``nchan``,
            ``df_hz``, ``flagged_fraction``, ``skipped`` (lines of the input
            list without data), ``bytes_on_disk``, ``seconds`` and the
            ``resources`` used; with *continuum*, ``contsub_orders``; with
            *stats_line*, ``statistics``.
        """
        from .resources import (available_memory, chunk_bytes, plan_gpu_batch,
                                plan_workers, reader_mode as choose_reader_mode,
                                shared_memory_free)
        if not output or central_freq is None:
            raise TypeError("run() needs the output path and central_freq")
        if output_format is None:
            output_format = "ms" if output.rstrip("/").lower().endswith(".ms") else "zarr"
        if output_format not in ("zarr", "ms"):
            raise ValueError("output_format must be 'zarr' or 'ms'")
        use_zarr = output_format == "zarr"
        if device not in ("cpu", "gpu"):
            raise ValueError("device must be 'cpu' or 'gpu'")
        n_ms = len(self.ms_list)
        plan = plan_workers(n_ms, use_zarr, n_readers, n_writers, n_threads)
        n_readers, n_writers = plan.n_readers, plan.n_writers
        out_queue = max(3, n_writers + 1)
        _reset_clock()
        t_start = time.perf_counter()
        dask.config.set(scheduler="synchronous")

        compute.set_threads(plan.n_threads)
        use_gpu = device == "gpu"
        if use_gpu:
            compute.check_gpu()
            gpu_batch = int(gpu_batch) if gpu_batch is not None else None
        chunk_queue = n_readers + 1
        self.data_column = data_column
        from .frames import TARGET_FRAMES
        frame = None if frame is None else str(frame).upper()
        if frame not in TARGET_FRAMES:
            raise ValueError(f"frame must be the frame of the redshifts, one of "
                             f"{TARGET_FRAMES} (default 'LSRK'), not {frame!r}: the "
                             f"inputs are always brought to it")
        self.frame = frame

        final = output
        log(f"ViSta {__version__} · {n_ms} MS from {os.path.basename(self.input_file)} "
            f"→ {final} ({'Zarr' if use_zarr else 'MS'}) · "
            + ("GPU" if use_gpu else "CPU") + f" · {plan.describe()}")

        # The readers start now, so that they import numpy / casacore while
        # the setup runs here; their jobs arrive once the setup is done.  On
        # the GPU (the queue must hold a whole batch) and with a small
        # /dev/shm they start after the setup, which tells how large the
        # chunks are.
        shm = shared_memory_free()
        early = not use_gpu and (shm is None or shm >= 4 * 2**30)
        readers = ReaderPool("process", n_readers, chunk_queue) if early else None
        try:
            output, subtables, ant_offsets = self._setup(
                final, scratch_dir, central_freq, nchan_out,
                velocity_range_kms, channel_width_hz)
            nchan_in = max(sum(len(f) for f in m.freq) for m in self._meta)
            rows_in = min(self.chunk_rows, max(1000, (512 * 2**20) // max(1, nchan_in * 16)))
            bytes_in = chunk_bytes(rows_in, nchan_in, 4)
            bytes_out = chunk_bytes(self.chunk_rows, self._grid.nchan, 4)
            if use_gpu:
                free_gpu = None
                try:
                    from . import ms_ops
                    free_gpu = int(ms_ops.gpu_memory_info()[0])
                except Exception:
                    pass
                gpu_batch = plan_gpu_batch(gpu_batch, bytes_in, bytes_out, free_gpu,
                                           available_memory(), n_writers)
                chunk_queue = max(chunk_queue, gpu_batch)   # a whole batch must fit
            if readers is None:
                mode = choose_reader_mode(n_readers, chunk_queue, bytes_in)
                if mode == "thread":
                    log(f"/dev/shm has {fmt_size(shm or 0)} free: the readers run as "
                        f"threads (their reads take turns)")
                readers = ReaderPool(mode, n_readers, chunk_queue)
            if use_gpu:
                log(f"GPU ({compute.gpu_memory_gb():.0f} GB): up to {gpu_batch} chunks "
                    f"per launch")
            detail(f"chunk_rows={self.chunk_rows}  reader_mode={readers.mode}  "
                   f"chunk_queue={chunk_queue}  out_queue={out_queue}")
            inline = None
            if continuum is not None or stats_line is not None:
                from .extract.inline import InlineExtraction
                inline = InlineExtraction(
                    self.input_file, self._grid.freq, final, line, continuum,
                    stats_line, stats_continuum, binning, max_amplitude,
                    verbose=bool(self.verbose))
                log(f"During the stacking: {inline.describe()}")
            elif line is not None or stats_continuum is not None:
                warn("line / stats_continuum given without continuum or stats_line: "
                     "nothing to do during the stacking")
            store = None
            if use_zarr:
                store = ZarrStackWriter(output, compress=True, attrs={
                    "vista_version": __version__, "central_freq_hz": float(central_freq),
                    "channel_width_hz": float(self._grid.df), "nchan": int(self._grid.nchan),
                    "n_ms": n_ms, "input_list": os.path.abspath(self.input_file),
                    "ms_list": list(self.ms_list), "z_list": [float(z) for z in self.z_list],
                    "frame": self.frame})
        except BaseException:
            if readers is not None:
                readers.stop()
            raise
        jobs = self._make_jobs(ant_offsets)
        readers.submit(jobs)

        run = _Run(output=output, jobs=jobs, readers=readers,
                   out_q=Queue(maxsize=out_queue), store=store, n_writers=n_writers,
                   gpu_batch=gpu_batch or 1, inline=inline)
        if use_gpu:
            run.gpu = compute.GpuEngine(out_queue + gpu_batch + n_writers + 1,
                                        self.chunk_rows, self._grid.nchan)
        run.t_loop = time.perf_counter()
        log(f"Setup done ({subtables['ANTENNA'].sizes['row']} antennas in total); "
            f"stacking {n_ms} MS...")
        self._stack(run)
        if (store is not None and not store.parts) or not os.path.isdir(output):
            raise RuntimeError(
                f"The stack '{output}' was not written: no MS gave usable data "
                f"(see the warnings above for the field, SPW and coverage of each MS)")

        extra = {}
        if inline is not None:
            extra = inline.close(final)
            rec = inline.contsub_record()
            if rec is not None and store is not None:
                store.set_record("contsub", rec)
        if store is not None:
            store.write_subtables(subtables)
            store.close()
        else:
            ms_output.write_subtables(output, subtables)
            if inline is not None and inline.contsub_record() is not None:
                from .stackio import MSStack
                stack = MSStack(output, writable=True)
                try:
                    stack.set_record("contsub", inline.contsub_record())
                finally:
                    stack.close()
        gc.collect()          # dask-ms keeps tables cached: close them now
        if scratch_dir is not None and output != final:
            log(f"Moving the output from the scratch directory to {final}...")
            shutil.move(output, final)
        summary = self._log_summary(run, final, output_format, t_start)
        summary["resources"] = {"cores": plan.cores, "n_readers": n_readers,
                                "n_writers": n_writers, "n_threads": plan.n_threads,
                                "reader_mode": readers.mode,
                                "gpu_batch": gpu_batch if use_gpu else None}
        summary.update(extra)
        return summary

    def _setup(self, final, scratch_dir, central_freq, nchan_out,
               velocity_range_kms, channel_width_hz):
        """Output path, spectral grid and output subtables.

        Returns
        -------
        (str, dict, list of int)
            Where the stack is written (in *scratch_dir* if given), the
            subtables to write at the end, and the antenna offset of every
            entry in the merged ANTENNA table.
        """
        output = final
        if scratch_dir is not None:
            os.makedirs(scratch_dir, exist_ok=True)
            output = os.path.join(scratch_dir, os.path.basename(final.rstrip("/")))
            log(f"Writing to the scratch directory {scratch_dir}, "
                f"moved to {final} at the end")
        for path in {output, final}:
            if os.path.exists(path):
                detail(f"Removing existing output: {path}")
                shutil.rmtree(path)

        self._compute_per_ms_grids(central_freq, nchan_out=nchan_out,
                                   velocity_range_kms=velocity_range_kms,
                                   channel_width_hz=channel_width_hz)
        self._log_grid(central_freq)
        subtables, ant_offsets = ms_output.build_subtables(
            self.ms_list, self.z_list, self._grid.freq, self._grid.df, central_freq)
        return output, subtables, ant_offsets

    # ---- the three stages ---------------------------------------------------------

    def _stack(self, run):
        """Readers, kernel and writers, running at the same time.

        The readers are already working; every writer gets its own thread,
        the kernel runs here.  Errors of any of them are re-raised here, after
        everything has stopped.
        """
        writers = [threading.Thread(target=self._writer_loop, args=(run,),
                                    name=f"vista-writer-{k}", daemon=True)
                   for k in range(run.n_writers)]
        for w in writers:
            w.start()
        self._compute_loop(run)
        for w in writers:
            w.join()
        run.t_loop_end = time.perf_counter()
        if run.gpu is not None:
            run.gpu.close()
        run.readers.stop()
        if run.errors:
            raise RuntimeError(f"Pipeline failed: {run.errors[0]!r}") from run.errors[0]

    def _compute_loop(self, run):
        """Main thread: take the reader messages and run the kernel on the chunks.

        On the CPU one chunk at a time; on the GPU also the chunks already
        waiting, up to ``gpu_batch``, without waiting for more.
        """
        messages = run.readers.messages
        n_done = 0
        try:
            while n_done < run.readers.n_readers:
                msg = run.get(messages)
                if msg[0] == "done":
                    n_done += 1
                if not self._handle(run, msg):
                    continue
                if run.gpu is None:
                    self._compute_cpu(run, msg[2], msg[3])
                    continue
                batch = [(msg[2], msg[3])]
                while len(batch) < run.gpu_batch:
                    try:
                        more = messages.get_nowait()
                    except Empty:
                        break
                    if more[0] == "done":
                        n_done += 1
                    if self._handle(run, more):
                        batch.append((more[2], more[3]))
                self._compute_gpu(run, batch)
            for _ in range(run.n_writers):
                run.put(run.out_q, _DONE)
        except _Abort:
            pass
        except Exception as e:
            run.fail(e)
        finally:
            if run.abort.is_set():
                run.readers.discard_queued()

    def _handle(self, run, msg):
        """Bookkeeping of a reader message; True if it is a chunk to compute."""
        kind, reader_id, index, obj, seconds = msg
        run.t_read += seconds
        if run.t_first is None and kind in ("info", "chunk"):
            run.t_first = time.perf_counter() - run.t_loop
        if kind == "chunk":
            return True
        if kind == "error":
            raise RuntimeError(f"reader {reader_id} failed:\n{obj}")
        if kind == "skip":
            job = run.jobs[index]
            run.skipped[index] = obj
            warn(f"{job.tag} {job.name} skipped: {obj}")
        elif kind == "info":
            run.infos[index] = obj
            if run.store is not None:       # its group in the store, before its chunks
                run.store.add_partition(index, obj.nrows, obj.chunk_rows,
                                        len(run.jobs[index].freq_new), obj.ncorr,
                                        extra=run.inline.extra_columns
                                        if run.inline is not None else ())
            self._log_entry(run.jobs[index], obj)
        return False

    def _compute_cpu(self, run, index, chunk):
        """One chunk on the CPU, then to the writer."""
        vis, flag, segments = run.readers.arrays(chunk)
        try:
            t0 = time.perf_counter()
            out = compute.run_cpu(vis, flag, chunk.uvw, run.jobs[index], run.infos[index],
                                  chunk.segments)
            run.t_compute += time.perf_counter() - t0
        finally:
            del vis, flag
            release(segments)
        self._emit(run, index, chunk, *out)

    def _compute_gpu(self, run, batch):
        """Several chunks, ``[(index, chunk), ...]``, in one GPU launch, then to the writer."""
        gpu = run.gpu
        inputs, segments, slots, outs = [], [], [], []
        try:
            for _, chunk in batch:
                vis, flag, segs = run.readers.arrays(chunk)
                inputs.append((vis, flag))
                segments += segs
            del vis, flag
            for n, ncorr in [(v.shape[0], v.shape[2]) for v, _ in inputs]:
                slot = gpu.take(n, ncorr, run.get)    # waiting here = waiting for the writer
                slots.append(slot)
                outs.append(gpu.outputs(slot, n, ncorr))
            # one block per chunk, or per run of rows with the same Doppler factor
            blk = {"vis": [], "flag": [], "uvw": [], "jobs": [], "infos": [],
                   "outs": [], "factors": []}
            for (index, chunk), (v, f), out in zip(batch, inputs, outs):
                for a, b, k in chunk.segments or [(0, v.shape[0], None)]:
                    blk["vis"].append(v[a:b])
                    blk["flag"].append(f[a:b])
                    blk["uvw"].append(chunk.uvw[a:b])
                    blk["jobs"].append(run.jobs[index])
                    blk["infos"].append(run.infos[index])
                    blk["outs"].append(tuple(o[a:b] for o in out))
                    blk["factors"].append(k)
            t0 = time.perf_counter()
            gpu.run(blk["vis"], blk["flag"], blk["uvw"], blk["jobs"], blk["infos"],
                    blk["outs"], factors=blk["factors"])
            del blk
            run.t_compute += time.perf_counter() - t0
        except BaseException:
            for slot in slots:                        # give back the buffers taken
                gpu.give_back(slot)
            raise
        finally:
            del inputs
            release(segments)
        for (index, chunk), out, slot in zip(batch, outs, slots):
            self._emit(run, index, chunk, *out, slot=slot)

    def _emit(self, run, index, chunk, vis_out, flag_out, uvw_out, slot=None):
        """Pack one processed chunk as output rows and hand it to the writers.

        The antennas are shifted to the merged ANTENNA table, DATA_DESC_ID is
        the line of the entry, FIELD_ID is 0, WEIGHT is multiplied by R and
        SIGMA by 1 / sqrt(R).
        """
        job, info = run.jobs[index], run.infos[index]
        n = vis_out.shape[0]
        columns = {
            "DATA": vis_out, "FLAG": flag_out, "UVW": uvw_out,
            "ANTENNA1": chunk.ant1 + job.ant_offset,
            "ANTENNA2": chunk.ant2 + job.ant_offset,
            "DATA_DESC_ID": np.full(n, index, dtype=np.int32),
            "FIELD_ID": np.zeros(n, dtype=np.int32),
        }
        for col, arr in chunk.row_columns.items():
            # in float64, rounded once to the type of the column
            if col == "WEIGHT":
                arr = (arr.astype(np.float64) * info.weight_scale).astype(arr.dtype)
            elif col == "SIGMA":
                arr = (arr.astype(np.float64) * info.sigma_scale).astype(arr.dtype)
            columns[col] = arr
        run.rows += n
        run.bytes += vis_out.nbytes + flag_out.nbytes + uvw_out.nbytes
        run.flagged += int(np.count_nonzero(flag_out))
        run.values += flag_out.size
        if run.inline is None:
            run.put(run.out_q, _Processed(columns, index, chunk.row0, slot, chunk.last))
            return
        if slot is not None:
            # a GPU buffer: copy the chunk and give the buffer back at once,
            # since the chunk may wait here (BIC sample) for a while
            for col in ("DATA", "FLAG", "UVW"):
                columns[col] = np.array(columns[col])
            run.gpu.give_back(slot)
        t0 = time.perf_counter()
        ready = run.inline.process(index, chunk.row0, columns, chunk.last)
        run.t_compute += time.perf_counter() - t0
        for row0, cols, last in ready:
            run.put(run.out_q, _Processed(cols, index, row0, None, last))

    def _writer_loop(self, run):
        """A writer thread: write processed chunks until the end of the stream.

        Zarr: ``n_writers`` such threads, each writing whole chunks at their
        place in the store (different files, no lock).  MS: one thread,
        appending every chunk to the main table.
        """
        table = ms_output.MsWriter(run.output) if run.store is None else None
        try:
            while True:
                item = run.get(run.out_q)
                if item is _DONE:
                    break
                t0 = time.perf_counter()
                if table is not None:
                    table.append(item.columns)
                else:
                    run.store.write(item.index, item.row0, item.columns)
                self._written(run, item, time.perf_counter() - t0)
            if table is not None:
                t0 = time.perf_counter()
                table.close()
                run.t_write += time.perf_counter() - t0
        except _Abort:
            pass
        except Exception as e:
            run.fail(e)
        finally:
            if table is not None:
                table.abandon()

    def _written(self, run, item, seconds):
        """Bookkeeping after a chunk is written (any writer thread)."""
        if item.slot is not None:
            run.gpu.give_back(item.slot)            # buffer free again
        n_ms = len(run.jobs)
        with run.lock:
            run.t_write += seconds
            if not item.last:
                return
            run.ms_done += 1
            k = run.ms_done
        step = max(1, n_ms // 10) if n_ms >= 20 else None
        if step and k % step == 0 and k < n_ms:
            elapsed = time.perf_counter() - run.t_loop
            log(f"Stacking: {k}/{n_ms} MS written ({100 * k / n_ms:.0f}%), "
                f"about {fmt_time(elapsed / k * (n_ms - k))} left")

    # ---- log -----------------------------------------------------------------------

    def _log_grid(self, central_freq):
        """The output grid, in two lines, and how many windows were slid."""
        g, n_ms = self._grid, len(self.ms_list)
        dv = g.df / central_freq * C_LIGHT_KMS
        log(f"Output grid: {g.nchan} channels × {g.df / 1e3:.1f} kHz ({dv:.2f} km/s), "
            f"±{0.5 * g.nchan * dv:.0f} km/s around {central_freq / 1e9:.4f} GHz")
        log(f"  ({g.width_note}; {g.nchan_note})")
        n_shift = sum(1 for s in g.shift if s)
        if n_shift:
            log(f"  {n_shift} of {n_ms} MS do not cover the whole window around "
                f"{central_freq / 1e9:.4f} GHz: their window is moved to the edge "
                f"of their band (shift in the lines below), so their channel "
                f"frequencies differ from the others")

    def _log_entry(self, job, info):
        """One line per entry, what it contributes to the stack, then its warnings."""
        rows = f"{info.nrows:,} rows"
        if info.rows_all is not None and info.rows_all != info.nrows:
            rows = f"{info.nrows:,}/{info.rows_all:,} rows ({', '.join(info.filters)})"
        cov = 100.0 * info.n_covered / max(1, len(job.freq_new))
        spw = f"{info.n_spw} SPW" + ("s" if info.n_spw > 1 else "")
        line = (f"{job.tag} {job.name}  z={job.z:.4f}  {rows}  {info.data_column}  "
                f"{spw}, {info.nchan} ch  covers {cov:.0f}% of the grid  "
                f"offset {info.offset_arcsec:.1f}\"")
        shift = self._grid.shift[job.index]
        if shift:
            line += f"  window {shift / 1e6:+.1f} MHz"
        log(line)
        if job.frame is not None:
            from .frames import velocity_kms
            detail(f"    frame {job.frame.input_frame} -> {job.frame.target_frame}: Earth's "
                   f"motion removed, {velocity_kms(job.frame.k_min):+.3f} .. "
                   f"{velocity_kms(job.frame.k_max):+.3f} km/s")
        detail(f"    weight scale R={info.weight_scale:.4f}  chunk={info.chunk_rows} rows  "
               f"ncorr={info.ncorr}  primary beam FWHM="
               + (f"{info.pb_fwhm_arcsec:.1f}\"" if info.pb_fwhm_arcsec else "n/a"))
        for w in info.warnings:
            warn(f"{job.tag} {job.name}: {w}")

    def _log_summary(self, run, final, output_format, t_start):
        """Skipped entries, timing (with verbose='debug') and a summary;
        returns the run summary."""
        n_ms = len(run.jobs)
        t_end = time.perf_counter()
        total = t_end - t_start
        flagged = run.flagged / run.values if run.values else 0.0
        n_spw = n_ms - len(run.skipped)
        if run.skipped:
            names = ", ".join(run.jobs[i].name for i in sorted(run.skipped))
            warn(f"{len(run.skipped)} of {n_ms} MS skipped, no data in the output: {names}")
        if _logger.isEnabledFor(logging.DEBUG):
            loop = run.t_loop_end - run.t_loop
            busy = {"reading": run.t_read / run.readers.n_readers,
                    "computing": run.t_compute, "writing": run.t_write / run.n_writers}
            limit = max(busy, key=busy.get)
            share = busy[limit] / max(loop, 1e-9)
            detail(f"Timing: setup {fmt_time(run.t_loop - t_start)} · stacking "
                f"{fmt_time(loop)} · subtables and closing "
                f"{fmt_time(t_end - run.t_loop_end)}")
            detail(f"  during the stacking (overlapping): reading "
                f"{fmt_time(busy['reading'])} per reader · computing "
                f"{fmt_time(busy['computing'])} · writing {fmt_time(busy['writing'])}"
                + (" per writer" if run.n_writers > 1 else "")
                + (f"  → limited by {limit} ({100 * share:.0f}% busy)" if share >= 0.5 else
                   f"  → no stage busy for half of the time: the first data arrived "
                   f"after {fmt_time(run.t_first or 0.0)} (start of the readers)"))
            if run.inline is not None:
                detail(f"  of the computing: continuum subtraction "
                    f"{fmt_time(run.inline.t_contsub)} · statistics "
                    f"{fmt_time(run.inline.t_stats)}")
            if run.gpu is not None and run.gpu.launches:
                detail(f"  GPU: {run.gpu.launches} launches, "
                    f"{run.gpu.chunks / run.gpu.launches:.1f} chunks per launch on average")
        on_disk = store_size(final)
        size = (f"{fmt_size(run.bytes)} of visibilities, {fmt_size(on_disk)} on disk"
                if output_format == "zarr" else fmt_size(on_disk))
        log(f"Done in {fmt_time(total)}: {final} · {n_spw} SPW × {self._grid.nchan} "
            f"channels · {run.rows:,} rows · {size} · "
            f"{100 * flagged:.1f}% of the output flagged (no data)")
        if output_format == "zarr":
            log("  (Zarr store: vista.export_ms() writes it as a Measurement Set)")
        return {"output": final, "format": output_format,
                "n_rows": run.rows, "n_spw": n_spw,
                "nchan": self._grid.nchan, "df_hz": float(self._grid.df),
                "flagged_fraction": flagged, "skipped": sorted(run.skipped),
                "bytes_on_disk": on_disk, "seconds": total}
