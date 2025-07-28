#!/usr/bin/env python3
"""
HamGNN 多维文档构建脚本
构建所有分支、版本和语言的文档
"""
import subprocess
import yaml
import os
import shutil
import sys
from pathlib import Path

def run_command(cmd, cwd=None):
    """运行命令并检查结果"""
    print(f"运行: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"错误: {result.stderr}")
        sys.exit(1)
    return result.stdout

def build_doc(branch, version, language, version_info):
    """构建单个文档"""
    print(f"\n构建: 分支={branch}, 版本={version}, 语言={language}")
    
    # 设置环境变量
    os.environ["build_all_docs"] = "true"
    os.environ["current_branch"] = branch
    os.environ["current_version"] = version
    os.environ["current_language"] = language
    os.environ["current_source_dir"] = version_info["source_dir"]
    
    # 设置页面根路径（从当前版本到根目录的相对路径）
    os.environ["pages_root"] = "../../.."
    
    # 设置 Sphinx 选项
    if language != "zh":
        os.environ['SPHINXOPTS'] = f"-D language='{language}'"
    else:
        os.environ['SPHINXOPTS'] = ""
    
    # 切换分支
    current_branch = run_command("git branch --show-current").strip()
    original_branch = current_branch  # 保存原始分支
    
    if current_branch != branch:
        # 保存当前更改
        run_command("git stash")
        run_command(f"git checkout {branch}")
    
    # 使用当前分支的配置文件（所有分支应该有相同的配置）
    # 不需要从其他分支 checkout 文件
    
    # 运行构建 - 使用临时目录避免覆盖
    temp_build_dir = f"_build_temp_{branch}_{version}_{language}"
    run_command(f"rm -rf {temp_build_dir}", cwd=".")
    
    # 如果是英文，需要使用 -D language=en 参数
    if language == "en":
        run_command(f"sphinx-build -b html -D language=en -d {temp_build_dir}/doctrees . {temp_build_dir}/html", cwd=".")
    else:
        run_command(f"sphinx-build -b html -d {temp_build_dir}/doctrees . {temp_build_dir}/html", cwd=".")
    
    # 返回原分支
    if original_branch != branch:
        run_command(f"git checkout {original_branch}")
        try:
            run_command("git stash pop")
        except:
            print("没有需要恢复的 stash")

def move_build(src, dst):
    """移动构建结果"""
    print(f"移动: {src} -> {dst}")
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dst_path.exists():
        shutil.rmtree(dst_path)
    shutil.move(src, dst)

def main():
    """主函数"""
    # 加载版本配置
    with open("versions.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # 创建输出目录
    output_base = Path("_build/all_docs")
    if output_base.exists():
        shutil.rmtree(output_base)
    
    # 构建所有组合
    for branch_key, branch_info in config["branches"].items():
        for version_key, version_info in branch_info["versions"].items():
            for lang_info in version_info["languages"]:
                lang_code = lang_info["code"]
                
                # 构建文档
                build_doc(branch_key, version_key, lang_code, version_info)
                
                # 移动到目标位置
                temp_build_dir = f"_build_temp_{branch_key}_{version_key}_{lang_code}"
                src = f"{temp_build_dir}/html"
                dst = output_base / branch_key / version_key / lang_code
                move_build(src, dst)
                
                # 清理临时目录
                shutil.rmtree(temp_build_dir)
    
    # 创建根目录的重定向页面
    defaults = config.get("defaults", {})
    default_branch = defaults['branch']
    default_version = defaults['version'] 
    default_language = defaults['language']
    
    # 获取默认版本的source_dir
    default_source_dir = config['branches'][default_branch]['versions'][default_version]['source_dir']
    default_url = f"{default_branch}/{default_version}/{default_language}/{default_source_dir}/index.html"
    
    index_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>HamGNN Documentation</title>
    <meta http-equiv="refresh" content="0; url={default_url}">
</head>
<body>
    <p>正在重定向到 <a href="{default_url}">默认文档</a>...</p>
</body>
</html>"""
    
    with open(output_base / "index.html", "w") as f:
        f.write(index_html)
    
    print(f"\n✅ 构建完成！文档位于: {output_base}")
    print(f"📁 目录结构:")
    for branch in config["branches"]:
        print(f"  └── {branch}/")
        for version in config["branches"][branch]["versions"]:
            print(f"      └── {version}/")
            for lang in config["branches"][branch]["versions"][version]["languages"]:
                print(f"          └── {lang['code']}/")

if __name__ == "__main__":
    # 切换到 docs 目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    main()