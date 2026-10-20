"""
Minimal example showing how to run ViSta.
"""
from vista import ViSta

# Initialise pipeline from input list
pipeline = ViSta(
    input_file="input_list.txt",
    chunk_rows=5000,   # number of baseline rows processed per chunk
    verbose=True,      # True: progress and warnings; "debug": every detail
)

# Run stacking
result = pipeline.run(
    ms_out="stacked_output.ms",
    central_freq=153.253e9,   # Hz -- rest-frame central frequency of the output grid
    nchan_out=1000,           # number of output channels
    device="cpu",             # "gpu" with a CUDA build of the kernel
    n_readers=2,              # MSs read in parallel
    timing=True,              # log how long reading, computing and writing took
)
print(result["ms_out"], result["n_rows"], "rows")
