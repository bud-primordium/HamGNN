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
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '../HamGNN_v_2_0/models/e3_layers.py']
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

# -- Custom event handler to skip specific headers -----------------------------

def remove_custom_header(app, what, name, obj, options, lines):
    """
    在 Sphinx 处理文档字符串时被调用，用于移除特定的文件头部。
    """
    header_signature = "Descripttion:"
    
    # 检查文档字符串的第一行是否包含我们的头部标记
    if lines and header_signature in lines[0]:
        # 如果是，就清空整个文档字符串列表，使其不被渲染
        lines.clear()

def setup(app):
    """
    将我们的自定义处理器注册到 Sphinx 的事件管理器中。
    """
    app.connect('autodoc-process-docstring', remove_custom_header)
