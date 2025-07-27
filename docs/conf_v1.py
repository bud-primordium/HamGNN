# docs/conf_v1.py - HamGNN v1.0 配置
import os
import sys

# 添加 v1.0 源代码路径
sys.path.insert(0, os.path.abspath('../HamGNN_v_1_0'))

# 导入主配置
exec(open('conf.py').read())

# 覆盖版本相关设置
project = 'HamGNN v1.0'
release = '1.0'
version = '1.0'

# 设置源文件目录
source_suffix = ['.rst', '.md']
master_doc = 'index'

# 更新 HTML 上下文
html_context.update({
    'current_version': 'v1.0',
    'versions': [
        {'name': 'v2.0', 'url': '/v2.0/'},
        {'name': 'v1.0', 'url': '/v1.0/'},
    ]
})

# 设置版本特定的构建目录
html_baseurl = '/v1.0/'