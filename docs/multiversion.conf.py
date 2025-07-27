# Sphinx build configuration file for sphinx-multiversion
# 定义哪些分支和标签需要构建文档

# 包含的分支模式（支持正则表达式）
smv_branch_whitelist = r'^(torchscript_export|chinese_annotated|main|master)$'

# 包含的标签模式（暂时不使用标签）
smv_tag_whitelist = r'^$'  # 不匹配任何标签

# 远程名称
smv_remote_whitelist = r'^(origin|upstream)$'

# 发布的分支（最新版本）
smv_released_pattern = r'^refs/heads/torchscript_export$'

# 输出格式
smv_outputdir_format = '{ref.name}'  # 使用分支名作为输出目录