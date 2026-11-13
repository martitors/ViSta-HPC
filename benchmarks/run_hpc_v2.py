"""ViSta-HPC v2 (branch ``extract_opt``) for the benchmark, step by step.

``--mode steps`` (the default) runs the three steps one after the other, as
the other two versions do, so that each can be timed:

1. ``stack``: the stack is written, nothing else;
2. ``contsub``: the continuum is subtracted on the stack;
3. ``extraction``: the stack is compressed and the line and the continuum are
   measured.

``--mode onepass`` is how the version is meant to be used: the continuum is
subtracted and the statistics are made while the stack is written
(``stack``), then the measurement (``extraction``).

It runs with ``verbose="debug"``, so that the log also holds where the time
of the stacking goes (reading, computing, writing), which ``bench.py`` reads.

Every step prints a ``@@stage <name>`` line, read by ``monitor.py``; the
results go to ``<output-name>_results.json`` and
``<output-name>_continuum_results.json``.
"""
import argparse

from vista import ViSta


def stage(name):
    print(f"@@stage {name}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-name", required=True)
    ap.add_argument("--rest-freq", type=float, required=True, help="GHz")
    ap.add_argument("--line-window", type=float, default=550.0)
    ap.add_argument("--velocity-range", type=float, default=900.0)
    ap.add_argument("--z-ref", type=float, default=None)
    ap.add_argument("--mode", choices=["steps", "onepass"], default="steps")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--output-format", choices=["zarr", "ms"], default="zarr")
    ap.add_argument("--readers", type=int, default=None)
    ap.add_argument("--writers", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--gpu-batch", type=int, default=None)
    a = ap.parse_args()

    def vista(continuum, extract):
        v = ViSta(a.input, output_name=a.output_name, device=a.device,
                  output_format=a.output_format, n_readers=a.readers, n_writers=a.writers,
                  n_threads=a.threads, gpu_batch=a.gpu_batch, verbose="debug")
        v.stack(line_freq=a.rest_freq, line_window=a.line_window,
                velocity_range=a.velocity_range, continuum=continuum)
        if extract:
            v.extract(z_ref=a.z_ref, flux_measure="gauss", figures_draw=False)
        return v

    if a.mode == "onepass":
        v = vista("subtract", extract=True)
        stage("stack")
        v.run(only="stack")
        stage("extraction")
        v.run(only="measure")
    else:
        stage("stack")
        vista("none", extract=False).run(only="stack")
        stage("contsub")
        vista("subtract", extract=False).run(only="continuum")
        stage("extraction")                  # no statistics yet: the stack is compressed first
        vista("subtract", extract=True).run(only="measure")


if __name__ == "__main__":
    main()
