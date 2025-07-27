#!/bin/bash
# 构建多版本文档

echo "=== 构建 HamGNN 多版本文档 ==="

# 切换到 docs 目录
cd "$(dirname "$0")"

# 检查是否使用 sphinx-multiversion
if command -v sphinx-multiversion &> /dev/null; then
    echo "使用 sphinx-multiversion 构建..."
    chmod +x build_multiversion.sh
    ./build_multiversion.sh
else
    echo "sphinx-multiversion 未安装，使用传统方式构建..."
    
    # 构建 v1.0
    echo "构建 v1.0 文档..."
    conda run -n hamgnn-docs sphinx-build -b html -c . source_v1 _build/html/v1.0
    
    # 构建 v2.0
    echo "构建 v2.0 文档..."
    conda run -n hamgnn-docs sphinx-build -b html -c . source_v2 _build/html/v2.0
    
    # 创建根目录重定向
    echo "创建根目录重定向..."
    cat > _build/html/index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>HamGNN Documentation</title>
    <meta http-equiv="refresh" content="0; url=./v2.0/">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background: #f8f9fa;
        }
        .container {
            text-align: center;
            padding: 2rem;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        a { color: #0969da; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <h1>HamGNN Documentation</h1>
        <p>Redirecting to <a href="./v2.0/">HamGNN v2.0 Documentation</a>...</p>
        <p>Or visit <a href="./v1.0/">HamGNN v1.0 Documentation</a></p>
    </div>
</body>
</html>
EOF
fi

echo "=== 文档构建完成！==="
echo "v1.0: $(pwd)/_build/html/v1.0/"
echo "v2.0: $(pwd)/_build/html/v2.0/"