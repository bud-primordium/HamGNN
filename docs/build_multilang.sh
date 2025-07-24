#!/bin/bash
# HamGNN 多版本多语言文档构建脚本

set -e  # 遇到错误立即退出

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 函数：打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 函数：构建特定版本和语言的文档
build_docs() {
    local version=$1
    local language=$2
    local source_dir=$3
    
    print_info "构建 ${version} - ${language} 文档..."
    
    # 创建临时配置文件
    cat > conf_temp.py << EOF
# 临时配置文件
import sys
sys.path.insert(0, '${source_dir}')

# 从主配置文件导入所有设置
from conf import *

# 覆盖版本和语言设置
version = '${version}'
release = '${version}'
language = '${language}'

# 调整路径
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '${source_dir}/models/e3_layers.py']
EOF
    
    # 构建文档
    sphinx-build -c . -b html . "_build/html/${language}/${version}"
    
    # 清理临时文件
    rm -f conf_temp.py
    
    print_success "完成 ${version} - ${language} 文档构建"
}

# 主程序
main() {
    print_info "开始构建 HamGNN 多版本多语言文档"
    
    # 清理旧的构建
    if [ "$1" == "clean" ]; then
        print_info "清理旧的构建文件..."
        rm -rf _build
    fi
    
    # 构建中文文档
    build_docs "v2.0" "zh_CN" "../HamGNN_v_2_0"
    build_docs "v1.0" "zh_CN" "../HamGNN_v_1_0"
    
    # 生成翻译模板
    if [ "$1" == "with-translation" ]; then
        print_info "生成翻译模板..."
        make gettext
        sphinx-intl update -p _build/gettext -l en
        
        # 如果存在翻译文件，构建英文文档
        if [ -d "locale/en" ]; then
            print_info "发现英文翻译，构建英文文档..."
            build_docs "v2.0" "en" "../HamGNN_v_2_0"
        else
            print_info "未发现英文翻译文件，跳过英文文档构建"
        fi
    fi
    
    # 创建根索引页面
    print_info "创建根索引页面..."
    cat > _build/html/index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>HamGNN Documentation</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .lang-section { margin: 20px 0; padding: 20px; background: #f5f5f5; border-radius: 8px; }
        h1 { color: #333; }
        h2 { color: #666; }
        a { color: #0066cc; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .version-list { margin-left: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>HamGNN Documentation</h1>
        
        <div class="lang-section">
            <h2>中文文档</h2>
            <div class="version-list">
                <ul>
                    <li><a href="zh_CN/v2.0/">v2.0 (最新版本)</a></li>
                    <li><a href="zh_CN/v1.0/">v1.0</a></li>
                </ul>
            </div>
        </div>
        
        <div class="lang-section">
            <h2>English Documentation</h2>
            <div class="version-list">
                <ul>
                    <li><a href="en/v2.0/">v2.0 (Latest)</a> <em>(Coming soon)</em></li>
                </ul>
            </div>
        </div>
    </div>
</body>
</html>
EOF
    
    print_success "文档构建完成！"
    print_info "文档位置: _build/html/"
}

# 执行主程序
main $@