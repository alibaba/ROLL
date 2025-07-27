#!/bin/bash
# Multi-Card LLM Generation Script
# 使用说明：./run_llm_gen_multicard.sh

pip install openai
cd /home/szhangfa/ROLL/Personality-Alignment/eval
set -e  # 遇到错误时退出

# 默认配置
DEFAULT_MODEL_PATH="/project/hdtaccuracy/models/base/Qwen3-8B/"
DEFAULT_NUM_GPUS=4
DEFAULT_PER_GPU_BATCH_SIZE=32  # 每张卡的批处理大小
DEFAULT_TOTAL_BATCH_SIZE=128   # 总批处理大小 (8 * 16)
DEFAULT_MAX_LIMIT=20000
DEFAULT_OUTPUT_FILE="qwen3_8b_lora_sft_results.jsonl"
DEFAULT_USE_ASYNC=true

# 可配置参数
MODEL_PATH=${MODEL_PATH:-$DEFAULT_MODEL_PATH}
LORA_PATH=${LORA_PATH:-"/project/hdtaccuracy/trains/sft/qwen3-8b-lora-sft"}
NUM_GPUS=${NUM_GPUS:-$DEFAULT_NUM_GPUS}
PER_GPU_BATCH_SIZE=${PER_GPU_BATCH_SIZE:-$DEFAULT_PER_GPU_BATCH_SIZE}
TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE:-$DEFAULT_TOTAL_BATCH_SIZE}
USE_ASYNC=${USE_ASYNC:-$DEFAULT_USE_ASYNC}
PROMPTS_FILE=${PROMPTS_FILE:-"/project/hdtaccuracy/Personality-Alignment/split_data_v5_filtered/filtered_dataset.jsonl"}
OUTPUT_FILE=${OUTPUT_FILE:-$DEFAULT_OUTPUT_FILE}
MAX_LIMIT=${MAX_LIMIT:-$DEFAULT_MAX_LIMIT}

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

echo_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

echo_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 使用说明
show_help() {
    echo "多卡并行推理使用方法："
    echo "  ./run_llm_gen_multicard.sh [选项]"
    echo ""
    echo "环境变量配置："
    echo "  MODEL_PATH         - 模型路径 (默认: $DEFAULT_MODEL_PATH)"
    echo "  LORA_PATH          - LoRA适配器路径 (可选)"
    echo "  NUM_GPUS           - 使用的GPU数量 (默认: $DEFAULT_NUM_GPUS)"
    echo "  PER_GPU_BATCH_SIZE - 每张GPU的批处理大小 (默认: $DEFAULT_PER_GPU_BATCH_SIZE)"
    echo "  TOTAL_BATCH_SIZE   - 总批处理大小 (默认: $DEFAULT_TOTAL_BATCH_SIZE)"
    echo "  USE_ASYNC          - 是否使用异步模式 (默认: $DEFAULT_USE_ASYNC)"
    echo "  PROMPTS_FILE       - 输入提示文件路径 (必需)"
    echo "  OUTPUT_FILE        - 输出文件路径 (默认: $DEFAULT_OUTPUT_FILE)"
    echo "  MAX_LIMIT          - 最大处理条数 (默认: $DEFAULT_MAX_LIMIT)"
    echo ""
    echo "使用示例："
    echo "  # 8卡并行，每卡batch_size=16"
    echo "  NUM_GPUS=8 PER_GPU_BATCH_SIZE=16 PROMPTS_FILE=data/prompts.jsonl ./run_llm_gen_multicard.sh"
    echo ""
    echo "  # 4卡并行，总batch_size=64"
    echo "  NUM_GPUS=4 TOTAL_BATCH_SIZE=64 PROMPTS_FILE=data/prompts.jsonl ./run_llm_gen_multicard.sh"
    echo ""
    echo "  # 使用异步模式提高吞吐量"
    echo "  NUM_GPUS=8 USE_ASYNC=true PROMPTS_FILE=data/prompts.jsonl ./run_llm_gen_multicard.sh"
    echo ""
    echo "  # 使用LoRA适配器"
    echo "  LORA_PATH=./lora_adapter NUM_GPUS=4 PROMPTS_FILE=data/prompts.jsonl ./run_llm_gen_multicard.sh"
    echo ""
    echo "推荐配置："
    echo "  - 对于Qwen2.5-7B: 每卡batch_size=16-32 (8张卡总共128-256)"
    echo "  - 对于Qwen2.5-14B: 每卡batch_size=8-16 (8张卡总共64-128)"
    echo "  - 使用异步模式可以提高20-30%的吞吐量"
    echo ""
    echo "选项："
    echo "  -h, --help         显示此帮助信息"
}

# 检查参数
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# 检查必需的参数
if [[ -z "$PROMPTS_FILE" ]]; then
    echo_error "PROMPTS_FILE 环境变量未设置！"
    echo_info "请设置输入文件路径，例如："
    echo "  PROMPTS_FILE=data/prompts.jsonl ./run_llm_gen_multicard.sh"
    exit 1
fi

# 检查输入文件是否存在
if [[ ! -f "$PROMPTS_FILE" ]]; then
    echo_error "输入文件不存在: $PROMPTS_FILE"
    exit 1
fi

# 检查LoRA路径（如果提供）
if [[ -n "$LORA_PATH" ]] && [[ ! -d "$LORA_PATH" ]]; then
    echo_error "LoRA路径不存在: $LORA_PATH"
    exit 1
fi

# 检查GPU数量
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [[ $NUM_GPUS -gt $AVAILABLE_GPUS ]]; then
    echo_warning "请求的GPU数量($NUM_GPUS)超过可用GPU数量($AVAILABLE_GPUS)，将使用所有可用GPU"
    NUM_GPUS=$AVAILABLE_GPUS
fi

# 显示GPU信息
echo_info "GPU状态信息："
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# 显示配置信息
echo_info "开始多卡LLM生成任务..."
echo_info "配置信息："
echo "  模型路径: $MODEL_PATH"
if [[ -n "$LORA_PATH" ]]; then
    echo "  LoRA路径: $LORA_PATH"
else
    echo "  LoRA路径: 未设置"
fi
echo "  GPU数量: $NUM_GPUS"
if [[ -n "$PER_GPU_BATCH_SIZE" ]]; then
    echo "  每GPU批处理大小: $PER_GPU_BATCH_SIZE"
    echo "  总批处理大小: $((PER_GPU_BATCH_SIZE * NUM_GPUS))"
else
    echo "  总批处理大小: $TOTAL_BATCH_SIZE"
    echo "  每GPU批处理大小: $((TOTAL_BATCH_SIZE / NUM_GPUS))"
fi
echo "  异步模式: $USE_ASYNC"
echo "  输入文件: $PROMPTS_FILE"
echo "  输出文件: $OUTPUT_FILE"
echo "  最大处理条数: $MAX_LIMIT"
echo ""

# 构建命令
CMD="python llm_gen_multicard.py"
CMD="$CMD --model_path \"$MODEL_PATH\""
CMD="$CMD --num_gpus $NUM_GPUS"
CMD="$CMD --prompts_file \"$PROMPTS_FILE\""
CMD="$CMD --output_file \"$OUTPUT_FILE\""
CMD="$CMD --max_limit $MAX_LIMIT"

if [[ -n "$LORA_PATH" ]]; then
    CMD="$CMD --lora_path \"$LORA_PATH\""
fi

if [[ -n "$PER_GPU_BATCH_SIZE" ]]; then
    CMD="$CMD --per_gpu_batch_size $PER_GPU_BATCH_SIZE"
else
    CMD="$CMD --inference_batch_size $TOTAL_BATCH_SIZE"
fi

if [[ "$USE_ASYNC" == "true" ]]; then
    CMD="$CMD --use_async"
fi

echo_info "执行命令: $CMD"
echo ""

# 检查Python和依赖
if ! command -v python &> /dev/null; then
    echo_error "Python未安装或未在PATH中"
    exit 1
fi

# 检查CUDA可用性
if ! python -c "import torch; print('CUDA available:', torch.cuda.is_available())"; then
    echo_error "CUDA不可用，请检查PyTorch安装"
    exit 1
fi

# 创建输出目录（如果不存在）
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo_info "创建输出目录: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# 记录开始时间
START_TIME=$(date +%s)

# 清理GPU内存
echo_info "清理GPU内存..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# 执行命令
echo_info "开始执行多卡推理..."
if eval $CMD; then
    # 计算执行时间
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))
    
    echo ""
    echo_success "多卡推理任务完成！"
    echo_info "执行时间: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
    echo_info "输出文件: $OUTPUT_FILE"
    
    # 显示输出文件信息
    if [[ -f "$OUTPUT_FILE" ]]; then
        LINE_COUNT=$(wc -l < "$OUTPUT_FILE")
        FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        echo_info "输出统计: $LINE_COUNT 行, $FILE_SIZE"
        
        # 计算吞吐量
        if [[ $DURATION -gt 0 ]]; then
            THROUGHPUT=$(echo "scale=2; $LINE_COUNT / $DURATION" | bc -l 2>/dev/null || echo "N/A")
            echo_info "处理速度: $THROUGHPUT samples/second"
        fi
    fi
    
    # 显示最终GPU状态
    echo_info "最终GPU状态："
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
    
else
    echo_error "多卡推理任务执行失败！"
    echo_info "最终GPU状态："
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
    exit 1
fi

# 清理GPU内存
echo_info "清理GPU内存..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
