# HamGNN 多语言文档系统使用指南

## 架构概述

HamGNN 文档系统现在支持三维切换：
- **分支 (Branch)**: chinese_annotated, torchscript_export
- **版本 (Version)**: v1.0, v2.0
- **语言 (Language)**: zh (中文), en (英文)

## 多语言支持工作流

### 1. 生成翻译文件

```bash
# 激活环境
conda activate hamgnn-docs

# 生成 .pot 模板和 .po 文件
python generate_translations.py
```

这会创建：
- `_build/gettext/*.pot` - 翻译模板文件
- `locale/en/LC_MESSAGES/*.po` - 待翻译的英文文件

### 2. 翻译 .po 文件

编辑 `locale/en/LC_MESSAGES/*.po` 文件，填写翻译：

```
msgid "原始中文文本"
msgstr "English translation"
```

**提示**: 可以使用翻译助手或其他工具批量翻译。

### 3. 构建多语言文档

```bash
# 构建所有版本和语言
python build_docs.py

# 仅测试英文构建
sphinx-build -b html -D language=en . _test_en
```

## 文件结构

```
docs/
├── locale/                 # 翻译文件
│   └── en/
│       └── LC_MESSAGES/
│           ├── source_v1/*.po
│           └── source_v2/*.po
├── _build/
│   ├── gettext/           # 翻译模板
│   └── all_docs/          # 构建输出
│       ├── chinese_annotated/
│       │   ├── v1.0/
│       │   │   ├── zh/
│       │   │   └── en/
│       │   └── v2.0/
│       │       ├── zh/
│       │       └── en/
│       └── torchscript_export/
│           └── ...
```

## 配置说明

### versions.yaml
定义了三维文档结构，每个版本都支持多语言：

```yaml
branches:
  chinese_annotated:
    versions:
      v2.0:
        languages:
          - code: "zh"
            name: "中文"
          - code: "en"
            name: "English"
```

### conf.py
包含 i18n 配置：

```python
# 多语言支持配置
locale_dirs = ['locale/']   # 翻译文件目录
gettext_compact = False     # 每个文档生成单独的 .pot 文件
```

### build_docs.py
支持根据语言参数构建：

```python
if language == "en":
    run_command(f"sphinx-build -b html -D language=en ...")
```

## 维护指南

1. **添加新内容后**：重新运行 `generate_translations.py` 更新翻译文件
2. **修改翻译**：直接编辑 .po 文件
3. **测试翻译效果**：使用测试构建命令验证

## 自动化建议

未来可以：
1. 使用 GitHub Actions 自动检测未翻译内容
2. 集成翻译 API 自动翻译新内容
3. 添加翻译进度跟踪

## 当前状态

- i18n 框架已搭建完成
- 支持中英双语切换
- 生成了所有翻译模板
- 完整翻译待完成（可使用翻译工具协助）