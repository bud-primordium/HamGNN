# docs/conf.py

import os
import sys

# -- Path setup --------------------------------------------------------------
# 将项目根目录添加到 sys.path，以便 Sphinx 能找到 HamGNN_v_2_0 模块
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'HamGNN'
copyright = 'HamGNN Team'
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
    'sphinx_copybutton',           # 代码复制按钮
    'sphinx.ext.graphviz',         # 生成类继承图
]

# MyST parser 配置
myst_enable_extensions = [
    "html_admonition",
    "dollarmath",  # 支持 LaTeX 数学公式 $...$ 和 $$...$$
]
myst_heading_anchors = 3

# Autodoc 配置 - 优化设置
autodoc_member_order = "bysource"
autosummary_generate = True
source_suffix = [".rst", ".md"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**/e3_layers.py']
language = 'zh_CN'

# 多语言支持配置
locale_dirs = ['locale/']   # 翻译文件目录
gettext_compact = False     # 每个文档生成单独的 .pot 文件
gettext_uuid = True         # 使用 UUID 追踪翻译
gettext_location = False    # 不在 .pot 文件中包含行号

# -- Options for HTML output -------------------------------------------------
# 使用 Furo 主题，现代化设计和更好的插件支持
html_theme = 'furo'
html_static_path = ['_static']

html_title = "HamGNN 中文文档"
html_logo = "_static/logo.png"
html_favicon = "_static/logo.png"
html_css_files = ["custom.css"]

html_copy_source = True
html_show_sourcelink = True
html_sourcelink_suffix = ''

# GitHub 源代码链接配置
html_context = {
    "display_github": True,
    "github_user": "bud-primordium",
    "github_repo": "HamGNN",
    "github_version": "torchscript_export",
    "conf_py_path": "/docs/",
}

html_theme_options = {
    "sidebar_hide_name": True,
    "top_of_page_buttons": ["view"],
    "navigation_with_keys": True,  # 允许键盘导航
    "announcement": None,
    # GitHub 集成
    "source_repository": "https://github.com/bud-primordium/HamGNN/",
    "source_branch": "torchscript_export",
    "source_directory": "docs/",
}

# 代码复制按钮配置
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_exclude = '.linenos, .gp'

# -- Intersphinx configuration -----------------------------------------------
# 配置跨项目文档链接
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'pytorch_lightning': ('https://lightning.ai/docs/pytorch/stable/', None),
    'e3nn': ('https://docs.e3nn.org/en/stable/', None),
    'torchmetrics': ('https://lightning.ai/docs/torchmetrics/stable/', None),
    'ase': ('https://wiki.fysik.dtu.dk/ase/', None),
}

# -- Multi-version configuration ----------------------------------------------
# 版本和语言配置（用于自定义多版本系统）
import os

# 从环境变量或默认值获取当前版本
CURRENT_VERSION = os.environ.get('HAMGNN_DOC_VERSION', 'v2.0')
CURRENT_LANGUAGE = os.environ.get('HAMGNN_DOC_LANGUAGE', 'zh_CN')

# 获取当前分支名（sphinx-multiversion会设置）
CURRENT_BRANCH = os.environ.get('SPHINX_MULTIVERSION_NAME', 'torchscript_export')

# 添加版本信息到模板上下文
html_context.update({
    'current_version': CURRENT_VERSION,
    'current_branch': CURRENT_BRANCH,
    'current_language': CURRENT_LANGUAGE,
})

# sphinx-multiversion 配置
# 检查是否在sphinx-multiversion环境中运行
is_multiversion = os.environ.get('SPHINX_MULTIVERSION_NAME') is not None

if is_multiversion:
    try:
        import sphinx_multiversion
        extensions.append('sphinx_multiversion')
        
        # 包含multiversion配置
        if os.path.exists('multiversion.conf.py'):
            exec(open('multiversion.conf.py').read())
        
        # 配置版本选择器模板
        if '_templates/multiversion' not in templates_path:
            templates_path.append('_templates/multiversion')
        
        # 更新HTML上下文以支持版本切换
        html_context['versions'] = [
            ('torchscript_export', '/torchscript_export/'),
            ('chinese_annotated', '/chinese_annotated/'),
        ]
        
        # Furo主题的版本切换器配置 - 动态生成当前分支选项
        branch_options = {
            'torchscript_export': 'TorchScript Export',
            'chinese_annotated': 'Chinese Annotated'
        }
        
        options_html = ''.join([
            f'<option value="/{branch}/v2.0/" {"selected" if branch == CURRENT_BRANCH else ""}>{label}</option>'
            for branch, label in branch_options.items()
        ])
        
        html_theme_options['announcement'] = \
            f"<b>分支:</b> <select onchange='window.location.href=this.value'>{options_html}</select>"
    except ImportError:
        print("Warning: sphinx-multiversion not installed")

# -- Custom event handler to skip specific headers -----------------------------

def remove_custom_header(app, what, name, obj, options, lines):
    """
    在 Sphinx 处理文档字符串时被调用，用于移除特定的文件头部。
    """
    # 定义一个元组，包含所有需要被识别和移除的头部“指纹”
    header_signatures = (
        "Descripttion:",
        "/*",
        "@Author:"
    )
    
    if not lines:
        return

    # 检查前几行内容是否包含任何一个“指纹”
    # 我们将前5行拼接起来检查，以应对各种格式
    docstring_head = "".join(lines[:5])
    for signature in header_signatures:
        if signature in docstring_head:
            lines.clear()
            # 一旦匹配成功，就清空并立即返回
            return

def setup(app):
    """
    将我们的自定义处理器注册到 Sphinx 的事件管理器中。
    """
    app.connect('autodoc-process-docstring', remove_custom_header)
