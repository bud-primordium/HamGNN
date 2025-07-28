#!/usr/bin/env python3
"""
生成和更新 HamGNN 文档的翻译文件
"""

import os
import subprocess
import sys

def run_command(cmd, cwd=None, ignore_errors=False):
    """运行命令并检查返回值"""
    print(f"运行命令: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0 and not ignore_errors:
        print(f"错误: {result.stderr}")
        sys.exit(1)
    print(result.stdout)
    if result.stderr and ignore_errors:
        print(f"警告: {result.stderr}")
    return result.stdout

def main():
    """主函数"""
    docs_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(docs_dir)
    
    print("=== 生成 HamGNN 文档翻译文件 ===")
    
    # 1. 清理旧的 gettext 文件
    print("\n1. 清理旧的 gettext 文件...")
    if os.path.exists("_build/gettext"):
        run_command("rm -rf _build/gettext")
    
    # 2. 生成 .pot 文件
    print("\n2. 生成 .pot 翻译模板文件...")
    run_command("make gettext")
    
    # 3. 创建/更新 .po 文件
    print("\n3. 创建/更新英文 .po 文件...")
    # 先清理旧的 locale 目录
    if os.path.exists("locale/en"):
        run_command("rm -rf locale/en")
    run_command("sphinx-intl update -p _build/gettext -l en")
    
    # 4. 统计需要翻译的条目
    print("\n4. 统计翻译情况...")
    po_files = []
    for root, dirs, files in os.walk("locale/en/LC_MESSAGES"):
        for file in files:
            if file.endswith(".po"):
                po_files.append(os.path.join(root, file))
    
    total_entries = 0
    for po_file in po_files:
        with open(po_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # 简单统计 msgid 的数量
            entries = content.count('msgid "')
            total_entries += entries
            print(f"  {po_file}: {entries} 条")
    
    print(f"\n总计需要翻译的条目: {total_entries}")
    
    # 5. 创建翻译说明文件
    readme_content = """# HamGNN 文档翻译指南

## 翻译工作流程

1. 翻译 `locale/en/LC_MESSAGES/*.po` 文件中的内容
2. 测试构建英文文档：`python build_docs.py`
3. 提交翻译更新

## PO 文件格式说明

```
msgid "原始中文文本"
msgstr "English translation"
```

## 翻译注意事项

1. 保持专业术语一致性
2. 代码示例中的注释也需要翻译
3. 保留 reStructuredText 格式标记
4. 数学公式和代码不翻译

## 批量翻译建议

可以将 .po 文件提供给翻译助手，使用以下提示词：

```
请帮我翻译这个 gettext .po 文件，从中文翻译到英文。
要求：
1. 保持专业术语准确
2. 保留所有格式标记
3. 只翻译 msgstr 部分
4. 代码和数学公式保持原样
```
"""
    
    with open("TRANSLATION_README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print(f"\n翻译文件生成完成！")
    print(f"   - .pot 模板文件: _build/gettext/")
    print(f"   - .po 翻译文件: locale/en/LC_MESSAGES/")
    print(f"   - 翻译指南: TRANSLATION_README.md")
    print(f"\n下一步：")
    print(f"   1. 编辑 locale/en/LC_MESSAGES/*.po 文件进行翻译")
    print(f"   2. 使用 'python build_docs.py' 构建多语言文档")

if __name__ == "__main__":
    main()