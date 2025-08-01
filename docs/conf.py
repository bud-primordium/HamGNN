# docs/conf.py

import os
import sys

# -- Path setup --------------------------------------------------------------
# 将项目根目录添加到 sys.path，以便 Sphinx 能找到 HamGNN 模块
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
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**/e3_layers.py', 'TRANSLATION_README.md']

# 多版本构建时的额外排除规则
build_all_docs = os.environ.get("build_all_docs")
if build_all_docs:
    current_version = os.environ.get("current_version", "v2.0")
    current_source_dir = os.environ.get("current_source_dir", "source_v2")
    
    # 设置主文档入口为版本特定的index
    master_doc = f"{current_source_dir}/index"
    
    # 构建 v2.0 时排除 v1.0 的内容和主index
    if current_version == "v2.0":
        exclude_patterns.extend(['source_v1/**', 'source_v1', 'index.rst'])
    # 构建 v1.0 时排除 v2.0 的内容和主index  
    elif current_version == "v1.0":
        exclude_patterns.extend(['source_v2/**', 'source_v2', 'index.rst'])
else:
    # 单版本构建时使用主index
    master_doc = 'index'

language = 'zh_CN'

# 多语言支持配置
locale_dirs = ['locale/']   # 翻译文件目录
gettext_compact = False     # 每个文档生成单独的 .pot 文件
gettext_uuid = True         # 使用 UUID 追踪翻译
gettext_location = False    # 不在 .pot 文件中包含行号

# -- Options for HTML output -------------------------------------------------
# 使用 RTD 主题，原生支持版本切换器
html_theme = 'sphinx_rtd_theme'
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
    "github_version": "chinese_annotated",
    "conf_py_path": "/docs/",
}

# RTD 主题配置 - 充分利用其特性
html_theme_options = {
    # 导航栏配置
    'navigation_depth': 4,  # 显示更深层级的导航
    'collapse_navigation': False,  # 默认展开导航
    'sticky_navigation': True,  # 滚动时导航栏固定
    'includehidden': True,
    'titles_only': False,  # 显示子标题
    
    # 显示配置
    'display_version': True,  # 显示版本号
    'prev_next_buttons_location': 'both',  # 上下页按钮显示在顶部和底部
    
    # 样式配置
    'style_nav_header_background': '#2980B9',  # 导航栏背景色
    'style_external_links': True,  # 外部链接添加图标
    
    # Logo 配置
    'logo_only': False,
    'display_version': True,
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

# 获取当前分支名
CURRENT_BRANCH = os.environ.get('SPHINX_MULTIVERSION_NAME', 'chinese_annotated')

# 添加版本信息到模板上下文
html_context.update({
    'current_version': CURRENT_VERSION,
    'current_branch': CURRENT_BRANCH,
    'current_language': CURRENT_LANGUAGE,
})

# 三维文档构建配置
# 检查是否在构建所有文档
build_all_docs = os.environ.get("build_all_docs")
pages_root = os.environ.get("pages_root", "")

if build_all_docs is not None:
    import yaml
    
    # 获取当前构建的三个维度
    current_branch = os.environ.get("current_branch", "chinese_annotated")
    current_version = os.environ.get("current_version", "v2.0") 
    current_language = os.environ.get("current_language", "zh")
    current_source_dir = os.environ.get("current_source_dir", "source_v2")
    
    
    # 更新版本和标题
    version = current_version
    release = current_version
    html_title = f"HamGNN {current_version} 中文文档"
        
    # 加载版本配置
    with open("versions.yaml", "r") as yaml_file:
        versions_config = yaml.safe_load(yaml_file)
    
    # 获取显示名称
    current_branch_display = versions_config['branches'][current_branch]['display_name']
    current_language_display = next(
        lang['name'] for lang in versions_config['branches'][current_branch]['versions'][current_version]['languages'] 
        if lang['code'] == current_language
    )
    
    # 构建 html_context
    html_context.update({
        'current_branch': current_branch,
        'current_version': current_version,
        'current_language': current_language,
        'current_branch_display': current_branch_display,
        'current_language_display': current_language_display,
        'branches': [],
        'versions': [],
        'languages': [],
    })
    
    # 构建分支列表
    for branch_key, branch_info in versions_config['branches'].items():
        # 获取目标分支中对应版本的source_dir（如果不存在则跳过）
        if current_version in branch_info['versions']:
            target_source_dir = branch_info['versions'][current_version]['source_dir']
            if branch_key == current_branch:
                branch_url = "index.html"  # 当前分支，不需要跳转
            else:
                # 分支切换：上升4级到根目录，然后进入目标分支
                branch_url = f"../../../../{branch_key}/{current_version}/{current_language}/{target_source_dir}/index.html"
            html_context['branches'].append([
                branch_key,
                branch_info['display_name'],
                branch_url
            ])
    
    # 构建版本列表
    if current_branch in versions_config['branches']:
        for version_key, version_info in versions_config['branches'][current_branch]['versions'].items():
            target_source_dir = version_info['source_dir']
            if version_key == current_version:
                version_url = "index.html"  # 当前版本，不需要跳转
            else:
                # 版本切换：上升3级到分支层，然后进入目标版本
                version_url = f"../../../{version_key}/{current_language}/{target_source_dir}/index.html"
            html_context['versions'].append([
                version_key,
                version_info['display_name'],
                version_url
            ])
    
    # 构建语言列表
    if current_branch in versions_config['branches'] and current_version in versions_config['branches'][current_branch]['versions']:
        current_version_source_dir = versions_config['branches'][current_branch]['versions'][current_version]['source_dir']
        for lang_info in versions_config['branches'][current_branch]['versions'][current_version]['languages']:
            if lang_info['code'] == current_language:
                lang_url = "index.html"  # 当前语言，不需要跳转
            else:
                # 语言切换：上升2级到版本层，然后进入目标语言
                lang_url = f"../../{lang_info['code']}/{current_version_source_dir}/index.html"
            html_context['languages'].append([
                lang_info['code'],
                lang_info['name'],
                lang_url
            ])

# -- Custom event handler to skip specific headers -----------------------------

def remove_custom_header(app, what, name, obj, options, lines):
    """
    在 Sphinx 处理文档字符串时被调用，用于移除特定的文件头部。
    """
    # 定义一个元组，包含所有需要被识别和移除的头部"指纹"
    header_signatures = (
        "Descripttion:",
        "/*",
        "@Author:"
    )
    
    if not lines:
        return

    # 检查前几行内容是否包含任何一个"指纹"
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