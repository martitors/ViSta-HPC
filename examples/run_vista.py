"""
Minimal example: stack a sample and measure its line.
"""
from vista import ViSta, export_ms

# input, output, how it runs
v = ViSta("input_list.txt",
          output_name="stacked",     # stacked.zarr, stacked_results.json, stacked_profile.png ...
          verbose=True)              # progress and results (default: warnings only)

# what makes the stack
v.stack(line_freq=345.79599,         # GHz, or "345.796GHz", "345796 MHz"; a list for more lines
        line_window=500,             # km/s: the line is within -500..+500 (or (lo, hi))
        velocity_range=1500,         # band of the stack, km/s (None: the widest of one dataset)
        continuum="subtract")        # 'subtract' | 'fit' | 'none'

# what is measured (without this call, run() only writes the stack)
v.extract(fwhm_range=(100, 600),     # limits on the width, km/s (lo = hi: fixed)
          theta_range=(0, 5))        # limits on the size of the source, arcsec

v.help()                             # every option: its value now, what it can be
result = v.run()
print(result)
ln = result.line.summary["line"]
print(f"F = {ln['flux']:.3f} ± {ln['flux_error']:.3f} ({ln['flux_method']}); "
      f"Gaussian {ln['flux_gauss']:.3f}, area {ln['flux_area']:.3f}")

# the measurement again, with other options, from the statistics (seconds); its
# files are stacked_area_* (the tag), so the first ones are kept
v.extract(fwhm_range=(300, 300), flux_measure="area")
print(v.run(only="measure", tag="area"))

# the stack as a Measurement Set, e.g. for imaging with tclean
export_ms("stacked.zarr", "stacked.ms")
