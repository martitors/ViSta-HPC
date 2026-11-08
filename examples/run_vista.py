"""
Minimal example showing how to run ViSta.
"""
from vista import ViSta, export_ms

# Initialise pipeline from input list
pipeline = ViSta(
    input_file="input_list.txt",
    chunk_rows=5000,   # number of baseline rows processed per chunk
    verbose=True,      # True: progress and warnings; "debug": every detail
)

# Run stacking
result = pipeline.run(
    output="stack.zarr",      # Zarr store, written in parallel (output_format="ms" for an MS)
    central_freq=153.253e9,   # Hz -- rest-frame central frequency of the output grid
    nchan_out=1000,           # number of output channels
    frame="LSRK",             # frame of the redshifts: "LSRK" (default) or "BARY"
    device="cpu",             # "gpu" with a CUDA build of the kernel
    n_readers=2,              # MSs read in parallel
    timing=True,              # log how long reading, computing and writing took
)
print(result["output"], result["n_rows"], "rows")

# Optional: a Measurement Set, e.g. for imaging with tclean
export_ms("stack.zarr", "stack.ms")
