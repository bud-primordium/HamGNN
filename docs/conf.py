# docs/conf.py

import os
import sys

# -- Path setup --------------------------------------------------------------
# 将项目根目录添加到 sys.path，以便 Sphinx 能找到 HamGNN_v_2_0 模块
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'HamGNN'
copyright = '2025, HamGNN Team'
author = 'HamGNN Team'
release = '2.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',          # 从 Docstrings 自动生成文档
    'sphinx.ext.napoleon',         # 解析 Google-style 和 NumPy-style Docstrings
    'sphinx.ext.viewcode',         # 在文档中添加源码链接
    'sphinx_autodoc_typehints',    # 将类型提示渲染到文档中
    'myst_parser',                 # 支持 Markdown 文件 (.md)
    'sphinx.ext.intersphinx',      # 链接到其他项目的文档 (Python, NumPy, PyTorch)
    'sphinx.ext.mathjax',          # 渲染 LaTeX 公式
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
language = 'zh_CN'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Intersphinx configuration -----------------------------------------------
# 配置跨项目文档链接
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'pytorch_lightning': ('https://lightning.ai/docs/pytorch/stable/', None),
}
