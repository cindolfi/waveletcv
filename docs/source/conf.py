# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'WaveletCV'
copyright = '2024, Christopher Indolfi'
author = 'Christopher Indolfi'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'breathe',
    'sphinx_readme',
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = []
cpp_maximum_signature_line_length = 20
toc_object_entries_show_parents = 'hide'

# -- Breathe -----------------------------------------------------------------
breathe_default_project = 'WaveletCV'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]
html_context = dict(
   display_github=True,
   github_user='chrismi',
   github_repo='wtcv',
   github_version='stable',
)
html_baseurl = 'https://wavletcv.readthedocs.io/en/latest'
readme_docs_url_type = 'html'
readme_src_files = 'README.rst'

html_theme_options = dict(
    navigation_depth = 8,
)

