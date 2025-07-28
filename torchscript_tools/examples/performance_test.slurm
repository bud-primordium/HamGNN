#!/bin/bash
#SBATCH --job-name=hamgnn_perf       # 作业名称
#SBATCH --output=perf_test_%j.out    # 标准输出文件
#SBATCH --error=perf_test_%j.err     # 标准错误文件
#SBATCH --time=02:00:00              # 最大运行时间
#SBATCH --partition=gpu              # 分区名称
#SBATCH --gres=gpu:1                 # 请求1个GPU
#SBATCH --mem=64G                    # 内存需求（性能测试需要更多内存）
#SBATCH --cpus-per-task=16           # CPU核心数

# HamGNN TorchScript 性能测试 SLURM 脚本示例
# 
# 使用方法:
#   1. 修改上面的 SLURM 参数以适应你的集群配置
#   2. 设置环境变量或修改脚本中的路径
#   3. 提交作业: sbatch performance_test.slurm

echo "========================================="
echo "HamGNN TorchScript 性能测试作业"
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_NODELIST"
echo "开始时间: $(date)"
echo "========================================="

# 激活conda环境（根据实际情况修改）
source /path/to/conda/bin/activate hamgnn-env

# 设置工作目录（根据实际情况修改）
cd /path/to/HamGNN

# 设置测试参数（可以通过环境变量覆盖）
CONFIG_PATH=${CONFIG_PATH:-"./configs/your_config.yaml"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"./checkpoints/your_model.ckpt"}
COMPILED_MODEL_PATH=${COMPILED_MODEL_PATH:-"./compiled_models/model.hamgnn.pt"}
# 注意：测试数据路径由配置文件中的dataset_params指定，无需单独设置

# 显示配置信息
echo "测试配置:"
echo "  CONFIG: $CONFIG_PATH"
echo "  CHECKPOINT: $CHECKPOINT_PATH"
echo "  COMPILED MODEL: $COMPILED_MODEL_PATH"
echo "  测试数据: 从配置文件中的dataset_params读取"

# 显示GPU信息
echo -e "\nGPU信息:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# 显示系统信息
echo -e "\n系统信息:"
echo "  CPU: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"
echo "  内存: $(free -h | grep Mem | awk '{print $2}')"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"

# 运行性能测试
echo -e "\n========================================="
echo "开始性能测试..."
echo "========================================="

# 导出环境变量供Python脚本使用
export CONFIG_PATH
export CHECKPOINT_PATH
export COMPILED_MODEL_PATH

# 执行性能测试
echo "注意：请确保在 HamGNN 根目录下运行，或使用正确的相对路径"
python torchscript_tools/examples/performance_test.py

# 检查测试结果
if [ $? -eq 0 ]; then
    echo -e "\n✅ 性能测试完成!"
else
    echo -e "\n❌ 性能测试失败!"
    exit 1
fi

# 可选：保存GPU使用情况
echo -e "\n测试后GPU状态:"
nvidia-smi

echo -e "\n========================================="
echo "结束时间: $(date)"
echo "总运行时间: $SECONDS 秒"
echo "========================================="