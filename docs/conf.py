# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'RapidAlign'
copyright = '2025, RapidAlign Authors'
author = 'RapidAlign Authors'
version = '0.1.0'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx_rtd_theme',
    'sphinx.ext.mathjax',
    'sphinx_copybutton',
    'nbsphinx',
    # 'sphinx_gallery.gen_gallery',  # Commenting out as it's causing errors
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = 'images/logo.svg'
html_favicon = 'images/logo.svg'
html_title = 'RapidAlign Documentation'

# -- MyST Parser configuration -----------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    # "linkify",  # Commenting out as it requires linkify-it-py
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# -- InterSphinx configuration -----------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# -- Autodoc configuration --------------------------------------------------
autodoc_typehints = 'description'
autodoc_member_order = 'bysource'
autoclass_content = 'both'

# -- nbsphinx configuration -------------------------------------------------
nbsphinx_execute = 'never'  # Don't execute the notebooks

# -- Sphinx-Gallery configuration -------------------------------------------
# Commenting out sphinx_gallery as it's causing errors
# sphinx_gallery_conf = {
#     'examples_dirs': '../examples',
#     'gallery_dirs': 'auto_examples',
#     'filename_pattern': '/example_',
#     'ignore_pattern': r'__init__\.py',
# }

# Add any paths that contain custom static files here
html_static_path = ['_static']

# Add custom CSS
html_css_files = [
    'css/custom.css',
]

# Setup the breathe extension
breathe_projects = {
    "RapidAlign": "./doxyoutput/xml"
}
breathe_default_project = "RapidAlign"