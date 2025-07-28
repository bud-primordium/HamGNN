#!/bin/bash
#SBATCH --job-name=hamgnn_compile    # 作业名称
#SBATCH --output=compile_%j.out      # 标准输出文件
#SBATCH --error=compile_%j.err       # 标准错误文件
#SBATCH --time=01:00:00              # 最大运行时间
#SBATCH --partition=gpu              # 分区名称
#SBATCH --gres=gpu:1                 # 请求1个GPU
#SBATCH --mem=32G                    # 内存需求
#SBATCH --cpus-per-task=8            # CPU核心数

# HamGNN TorchScript 编译 SLURM 脚本示例
# 
# 使用方法:
#   1. 修改上面的 SLURM 参数以适应你的集群配置
#   2. 修改下面的路径和参数
#   3. 提交作业: sbatch compile_hamgnn.slurm

echo "========================================="
echo "HamGNN TorchScript 编译作业"
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_NODELIST"
echo "开始时间: $(date)"
echo "========================================="

# 激活conda环境（根据实际情况修改）
source /path/to/conda/bin/activate hamgnn-env

# 设置工作目录（根据实际情况修改）
cd /path/to/HamGNN

# 设置相对路径
CONFIG_PATH="./configs/your_config.yaml"
CHECKPOINT_PATH="./checkpoints/your_model.ckpt"
OUTPUT_PATH="./compiled_models/model.hamgnn.pt"

# 创建输出目录
mkdir -p $(dirname $OUTPUT_PATH)

# 显示GPU信息
echo "GPU信息:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# 执行编译
echo -e "\n开始编译..."
hamgnn-compile \
    --config $CONFIG_PATH \
    --checkpoint $CHECKPOINT_PATH \
    --output $OUTPUT_PATH \
    --device cuda

# 检查编译结果
if [ $? -eq 0 ]; then
    echo -e "\n✅ 编译成功!"
    echo "输出文件:"
    ls -lh $OUTPUT_PATH*
    
    # 可选：立即运行性能测试
    echo -e "\n运行性能测试..."
    echo "注意：请确保在 HamGNN 根目录下运行，或使用正确的相对路径"
    python torchscript_tools/examples/performance_test.py
else
    echo -e "\n❌ 编译失败!"
    exit 1
fi

echo -e "\n========================================="
echo "结束时间: $(date)"
echo "========================================="