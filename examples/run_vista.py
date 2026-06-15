"""
Minimal example showing how to run ViSta.
"""
from vista import ViSta

# Initialise pipeline from input list
pipeline = ViSta(
    input_file="input_list.txt",
    chunk_rows=5000,   # number of baseline rows processed per chunk
    verbose=True,
)

# Run stacking
pipeline.run(
    ms_out="stacked_output.ms",
    central_freq=153.253e9,   # Hz -- rest-frame central frequency of the output grid
    nchan_out=1000,           # number of output channels
)
