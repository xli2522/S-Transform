"""Sphinx configuration for the s-transform documentation."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "s-transform"
copyright = "2021, Xiyuan Li"
author = "Xiyuan Li"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "alabaster"
