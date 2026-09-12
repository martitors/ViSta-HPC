from setuptools import setup, find_packages

setup(
    name="vista-hpc",
    version="2.0.0",
    description="HPC-optimised visibility-domain stacking for interferometric data",
    author="Martina Torsello",
    packages=find_packages(),
    package_data={"vista": ["ms_ops*.so"]},
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "dask>=2022.1",
        "dask-ms>=0.2.18",
        "xarray>=0.19",
        "scipy>=1.7",          # vista.extract: profile and uv-plane fitting
    ],
    extras_require={
        "plots": ["matplotlib>=3.4"],
        "cosmology": ["astropy>=5.0"],
    },
)
