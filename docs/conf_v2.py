# docs/conf_v2.py - HamGNN v2.0 配置
import os
import sys

# 添加 v2.0 源代码路径
sys.path.insert(0, os.path.abspath('../HamGNN_v_2_0'))

# 导入主配置
exec(open('conf.py').read())

# 覆盖版本相关设置
project = 'HamGNN v2.0'
release = '2.0'
version = '2.0'

# 设置源文件目录
source_suffix = ['.rst', '.md']
master_doc = 'index'

# 更新 HTML 上下文
html_context.update({
    'current_version': 'v2.0',
    'versions': [
        {'name': 'v2.0', 'url': '/v2.0/'},
        {'name': 'v1.0', 'url': '/v1.0/'},
    ]
})

# 设置版本特定的构建目录
html_baseurl = '/v2.0/'