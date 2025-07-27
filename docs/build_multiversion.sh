#!/bin/bash
# 使用 sphinx-multiversion 构建多分支文档

echo "=== 使用 sphinx-multiversion 构建文档 ==="

# 切换到 docs 目录
cd "$(dirname "$0")"

# 构建多分支文档
echo "构建多分支文档..."
conda run -n hamgnn-docs sphinx-multiversion . _build/html \
    -D smv_branch_whitelist="^(torchscript_export|chinese_annotated)$" \
    -D smv_tag_whitelist="^$" \
    -D smv_remote_whitelist="^(origin|upstream)$"

# 每个分支内部还需要构建 v1.0 和 v2.0
for branch in torchscript_export chinese_annotated; do
    if [ -d "_build/html/$branch" ]; then
        echo "为 $branch 分支构建版本特定文档..."
        
        # 构建 v1.0
        SPHINX_MULTIVERSION_NAME=$branch conda run -n hamgnn-docs sphinx-build -b html -c . source_v1 _build/html/$branch/v1.0
        
        # 构建 v2.0  
        SPHINX_MULTIVERSION_NAME=$branch conda run -n hamgnn-docs sphinx-build -b html -c . source_v2 _build/html/$branch/v2.0
        
        # 为分支创建版本选择页面
        cat > _build/html/$branch/index.html << EOF
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>HamGNN Documentation - $branch</title>
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
        <h1>HamGNN Documentation - $branch</h1>
        <p>Redirecting to <a href="./v2.0/">v2.0 Documentation</a>...</p>
        <p>Or visit <a href="./v1.0/">v1.0 Documentation</a></p>
    </div>
</body>
</html>
EOF
    fi
done

# 创建根目录索引
cat > _build/html/index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>HamGNN Documentation</title>
    <meta http-equiv="refresh" content="0; url=./torchscript_export/v2.0/">
</head>
<body>
    <p>Redirecting to <a href="./torchscript_export/v2.0/">HamGNN Documentation</a>...</p>
</body>
</html>
EOF

echo "=== 文档构建完成！==="
ls -la _build/html/