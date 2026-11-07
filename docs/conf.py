"""Sphinx configuration of the ViSta documentation (Read the Docs)."""
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "ViSta"
author = "Martina Torsello"
copyright = "2025-2026, Martina Torsello"
release = "2.5.0"
version = "2.5"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = ["_build", "_tools", "examples/pages.txt"]

myst_enable_extensions = ["colon_fence", "dollarmath", "deflist", "attrs_inline"]
myst_heading_anchors = 3

# the API pages import the package: the compiled kernel and the I/O libraries
# are not built or installed on Read the Docs, so they are mocked
autodoc_mock_imports = ["vista.ms_ops", "casacore", "casatools", "daskms", "dask",
                        "zarr", "numcodecs", "xarray", "matplotlib"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

html_theme = "sphinx_rtd_theme"
html_logo = "_static/logo.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "ViSta documentation"
html_theme_options = {
    "logo_only": False,
    "navigation_depth": 3,
    "collapse_navigation": False,
}
html_context = {
    "display_github": True,
    "github_user": "martitors",
    "github_repo": "ViSta-HPC",
    "github_version": "extract_opt",
    "conf_py_path": "/docs/",
}

copybutton_prompt_text = r"\$ |>>> "
copybutton_prompt_is_regexp = True
