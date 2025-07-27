# HamGNN 多版本文档系统

## 概述

HamGNN 使用 Sphinx + Furo 主题 + sphinx-multiversion 构建多分支多版本文档系统，支持不同分支（torchscript_export、chinese_annotated）和不同版本（v1.0、v2.0）的独立文档。

## 文档架构

### 三层结构
1. **分支层**：通过 sphinx-multiversion 管理不同分支
2. **版本层**：每个分支内部支持 v1.0 和 v2.0
3. **模块层**：每个版本包含多个模块的 API 文档

### 目录结构

```
docs/
├── conf.py                 # 主配置文件（支持sphinx-multiversion）
├── multiversion.conf.py    # sphinx-multiversion 配置
├── source_v1/              # v1.0 文档源文件
│   ├── index.rst           # v1.0 主页（含版本切换链接）
│   ├── data_processing/    # 数据处理模块（已拆分）
│   ├── gnn_core/          # GNN核心模块（已拆分）
│   └── model_components/   # 模型组件（已拆分）
├── source_v2/              # v2.0 文档源文件
│   └── (类似结构)
├── _templates/             # 自定义模板
├── _static/               # 静态资源
├── build_docs.sh          # 本地构建脚本
├── build_multiversion.sh  # sphinx-multiversion 构建脚本
└── environment.yml        # Conda 环境配置
```

## 核心特性

### 1. 多分支支持（sphinx-multiversion）
- 自动构建配置的分支：torchscript_export、chinese_annotated
- 提供专业的分支切换下拉菜单
- 每个分支独立构建和部署

### 2. 多版本支持（自定义实现）
- 每个分支内包含 v1.0 和 v2.0 两个版本
- 通过 index.rst 中的链接进行版本切换
- 版本间完全独立，互不干扰

### 3. Furo 主题特性
- 现代化的文档主题
- 内置深色模式支持
- 响应式设计
- 优秀的代码高亮
- 分支切换器集成在 announcement 栏

### 4. 功能配置
- **查看源代码**: 链接到 GitHub 仓库对应文件
- **深色模式**: 用户可切换
- **代码复制**: 一键复制代码块（sphinx-copybutton）
- **数学公式**: 支持 LaTeX 渲染（MathJax）
- **交叉引用**: 链接到 Python、PyTorch、e3nn 等官方文档
- **类继承图**: 通过 Graphviz 生成

## 构建指南

### 本地构建

1. 激活 Conda 环境：
```bash
conda activate hamgnn-docs
```

2. 运行构建脚本：
```bash
cd docs
./build_docs.sh
```

脚本会自动检测是否安装了 sphinx-multiversion：
- 已安装：使用 build_multiversion.sh 构建多分支文档
- 未安装：传统方式构建当前分支的两个版本

3. 本地预览：
```bash
# 如果使用了 sphinx-multiversion
python -m http.server 8000 -d _build/html

# 传统方式
python -m http.server 8000 -d _build/html/v2.0
```

### GitHub Actions 自动部署

文档会在推送到配置的分支时自动构建并部署到 GitHub Pages。

工作流程：
1. 获取所有分支的完整历史
2. 使用 sphinx-multiversion 构建多分支文档
3. 每个分支内部构建 v1.0 和 v2.0
4. 部署到 gh-pages 分支

配置文件：`.github/workflows/deploy-multiversion-docs.yml`

## 版本切换机制

### 分支切换
- 位置：页面顶部 announcement 栏的下拉菜单
- 实现：Furo 主题的 announcement 选项 + sphinx-multiversion
- 支持：torchscript_export ⇄ chinese_annotated

### 版本切换（v1.0 ⇄ v2.0）
- 位置：每个版本主页的提示框
- 实现：index.rst 中的相对链接
- 格式：`查看 v2.0 文档 <../v2.0/index.html>`

## 版本差异

### v1.0
- 基于 nequip 框架的早期版本
- 模块路径：`HamGNN_v_1_0.models.*`
- 主要特性：基础的等变图神经网络

### v2.0
- 全新的 Attention-KAN 架构
- 模块路径：`HamGNN_v_2_0.models.HamGNN.*`
- 主要特性：注意力机制、KAN 网络、优化的性能

## 配置说明

### sphinx-multiversion 配置（multiversion.conf.py）
```python
# 包含的分支
smv_branch_whitelist = r'^(torchscript_export|chinese_annotated|main|master)$'
# 不使用标签
smv_tag_whitelist = r'^$'
# 输出目录格式
smv_outputdir_format = '{ref.name}'
```

### 添加新分支
1. 更新 `multiversion.conf.py` 的 `smv_branch_whitelist`
2. 更新 `conf.py` 的分支选项
3. 更新 GitHub Actions 的分支触发器

### 添加新版本（如 v3.0）
1. 创建 `source_v3/` 目录结构
2. 更新 `build_multiversion.sh` 添加 v3.0 构建
3. 在各版本的 index.rst 中添加切换链接

## 故障排除

### sphinx-multiversion 相关
- 错误：`No matching branches found`
  - 确保本地有所有配置的分支
  - 运行 `git fetch --all`

### 版本切换器不工作
- 检查 index.rst 中的相对路径是否正确
- 确认构建输出目录结构

### API 文档不生成
- 确认 Python 路径设置正确
- 检查模块是否可以被导入
- 查看 sphinx-build 的错误输出

## 最佳实践

1. **提交前测试**：本地运行 build_docs.sh 确保文档能正确构建
2. **保持一致性**：两个版本的文档结构保持相似
3. **及时更新**：代码更新后同步更新对应的文档
4. **分支管理**：确保分支间的文档配置独立

## 未来扩展

1. **多语言支持**：通过 Sphinx i18n 机制添加国际化
2. **PDF 导出**：添加 LaTeX 构建支持
3. **搜索优化**：集成更强大的搜索功能
4. **API 版本对比**：自动生成版本间的差异文档