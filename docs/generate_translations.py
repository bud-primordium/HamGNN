#!/usr/bin/env python3
"""
生成和更新 HamGNN 文档的翻译文件

支持两种模式：
1. --clean: 清理并重新生成所有翻译文件（会丢失已有翻译）
2. 默认: 增量更新，保留已有翻译
"""

import os
import subprocess
import sys
import argparse
import shutil
from datetime import datetime

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

def backup_translations():
    """备份现有的翻译文件"""
    if os.path.exists("locale/en"):
        backup_name = f"locale_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"\n备份现有翻译到 {backup_name}...")
        shutil.copytree("locale/en", backup_name)
        print(f"备份完成！")
        return backup_name
    return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生成和更新 HamGNN 文档翻译文件')
    parser.add_argument('--clean', action='store_true', 
                        help='清理并重新生成所有翻译文件（警告：会丢失已有翻译）')
    parser.add_argument('--backup', action='store_true', 
                        help='在更新前备份现有翻译')
    args = parser.parse_args()
    
    docs_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(docs_dir)
    
    print("=== 生成 HamGNN 文档翻译文件 ===")
    
    # 备份处理
    if args.backup and os.path.exists("locale/en"):
        backup_translations()
    
    # 1. 清理旧的 gettext 文件
    print("\n1. 清理旧的 gettext 文件...")
    if os.path.exists("_build/gettext"):
        run_command("rm -rf _build/gettext")
    
    # 2. 生成 .pot 文件
    print("\n2. 生成 .pot 翻译模板文件...")
    run_command("make gettext")
    
    # 3. 创建/更新 .po 文件
    if args.clean:
        print("\n3. 清理模式：删除并重新创建翻译文件...")
        print("警告：这将删除所有已有的翻译！")
        if os.path.exists("locale/en"):
            run_command("rm -rf locale/en")
        run_command("sphinx-intl update -p _build/gettext -l en -j 1")
    else:
        print("\n3. 增量更新模式：保留已有翻译...")
        # sphinx-intl update 会自动合并已有翻译
        run_command("sphinx-intl update -p _build/gettext -l en -j 1")
    
    # 4. 使用 sphinx-intl stat 统计翻译情况
    print("\n4. 统计翻译情况...")
    print("\n使用 sphinx-intl stat 获取详细统计：")
    print("=" * 70)
    
    # 运行 sphinx-intl stat 命令
    stat_output = run_command("sphinx-intl stat -d locale -l en")
    
    # 解析 sphinx-intl stat 输出，计算总计
    import re
    total_translated = 0
    total_fuzzy = 0
    total_untranslated = 0
    
    # 匹配每行的统计信息
    stat_pattern = re.compile(r': (\d+) translated, (\d+) fuzzy, (\d+) untranslated\.')
    for line in stat_output.split('\n'):
        match = stat_pattern.search(line)
        if match:
            total_translated += int(match.group(1))
            total_fuzzy += int(match.group(2))
            total_untranslated += int(match.group(3))
    
    total_entries = total_translated + total_fuzzy + total_untranslated
    
    print("\n" + "=" * 70)
    print("总计统计：")
    print(f"  总条目数: {total_entries}")
    print(f"  已翻译: {total_translated} ({total_translated/total_entries*100:.1f}%)")
    print(f"  模糊匹配: {total_fuzzy} ({total_fuzzy/total_entries*100:.1f}%)")
    print(f"  未翻译: {total_untranslated} ({total_untranslated/total_entries*100:.1f}%)")
    print("=" * 70)
    
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
  1. 保持专业术语准确（如 Hamiltonian, irreps, tensor product, equivariant 等）
  2. 保留所有格式标记（如 :math:`...`, :py:class:`...`, **粗体**, `代码` 等）
  3. 只翻译 msgstr "" 部分，msgid 部分保持不变
  4. 数学公式、代码片段、文件路径保持原样
  5. 参数类型格式保持一致（如 "int", "float", "torch.Tensor" 等）

  特殊注意事项：
  - 空的 msgstr "" 需要填入对应的英文翻译
  - 保持 reStructuredText (rst) 格式的完整性
  - 注意跨行的长字符串要保持正确的引号格式
  - 专业术语翻译参考：
    - 不可约表示 → irreducible representations (irreps)
    - 张量积 → tensor product
    - 球谐函数 → spherical harmonics
    - 等变 → equivariant
    - 哈密顿量 → Hamiltonian
    - 基组 → basis set
    - 角动量 → angular momentum

  示例：
  #: 522371d761274a24bed68d1527daad4b
  msgid "将哈密顿量或重叠矩阵从原子轨道基组展开为球谐函数基组。"
  msgstr "Expand Hamiltonian or overlap matrices from atomic orbital basis sets to spherical harmonic basis sets."
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
    print(f"\n提示：")
    print(f"   - 使用 --backup 参数在更新前备份翻译")
    print(f"   - 使用 --clean 参数重新生成所有翻译文件（警告：会丢失已有翻译）")

if __name__ == "__main__":
    main()