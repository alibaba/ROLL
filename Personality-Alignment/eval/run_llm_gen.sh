#!/bin/bash
# LLM Generation Script
# 使用说明：./run_llm_gen.sh
pip install openai
cd /home/szhangfa/ROLL/Personality-Alignment/eval
set -e  # 遇到错误时退出

# 默认配置
DEFAULT_MODEL_PATH="/project/hdtaccuracy/models/base/Qwen3-8B/"
DEFAULT_DEVICE="auto"
DEFAULT_BATCH_SIZE=32  # 降低默认批处理大小以避免OOM
DEFAULT_MAX_LIMIT=20000
DEFAULT_OUTPUT_FILE="qwen_base_results.jsonl"

# 可配置参数
MODEL_PATH=${MODEL_PATH:-$DEFAULT_MODEL_PATH}
LORA_PATH=${LORA_PATH:-""}
DEVICE=${DEVICE:-$DEFAULT_DEVICE}
BATCH_SIZE=${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}
PROMPTS_FILE=${PROMPTS_FILE:-"/project/hdtaccuracy/Personality-Alignment/dialogue_dataset_all_v5_summarized.jsonl"}
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
    echo "使用方法："
    echo "  ./run_llm_gen.sh [选项]"
    echo ""
    echo "环境变量配置："
    echo "  MODEL_PATH      - 模型路径 (默认: $DEFAULT_MODEL_PATH)"
    echo "  LORA_PATH       - LoRA适配器路径 (可选)"
    echo "  DEVICE          - 设备类型 (默认: $DEFAULT_DEVICE)"
    echo "  BATCH_SIZE      - 批处理大小 (默认: $DEFAULT_BATCH_SIZE)"
    echo "  PROMPTS_FILE    - 输入提示文件路径 (必需)"
    echo "  OUTPUT_FILE     - 输出文件路径 (默认: $DEFAULT_OUTPUT_FILE)"
    echo "  MAX_LIMIT       - 最大处理条数 (默认: $DEFAULT_MAX_LIMIT)"
    echo ""
    echo "使用示例："
    echo "  # 基本使用"
    echo "  PROMPTS_FILE=data/prompts.jsonl ./run_llm_gen.sh"
    echo ""
    echo "  # 使用LoRA适配器"
    echo "  PROMPTS_FILE=data/prompts.jsonl LORA_PATH=./lora_adapter ./run_llm_gen.sh"
    echo ""
    echo "  # 自定义配置"
    echo "  MODEL_PATH=./my_model PROMPTS_FILE=data/prompts.jsonl BATCH_SIZE=4 ./run_llm_gen.sh"
    echo ""
    echo "选项："
    echo "  -h, --help      显示此帮助信息"
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
    echo "  PROMPTS_FILE=data/prompts.jsonl ./run_llm_gen.sh"
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

# 显示配置信息
echo_info "开始LLM生成任务..."
echo_info "配置信息："
echo "  模型路径: $MODEL_PATH"
if [[ -n "$LORA_PATH" ]]; then
    echo "  LoRA路径: $LORA_PATH"
else
    echo "  LoRA路径: 未设置"
fi
echo "  设备: $DEVICE"
echo "  批处理大小: $BATCH_SIZE"
echo "  输入文件: $PROMPTS_FILE"
echo "  输出文件: $OUTPUT_FILE"
echo "  最大处理条数: $MAX_LIMIT"
echo ""

# 构建命令
CMD="python llm_gen.py"
CMD="$CMD --model_path \"$MODEL_PATH\""
CMD="$CMD --device \"$DEVICE\""
CMD="$CMD --inference_batch_size $BATCH_SIZE"
CMD="$CMD --prompts_file \"$PROMPTS_FILE\""
CMD="$CMD --output_file \"$OUTPUT_FILE\""
CMD="$CMD --max_limit $MAX_LIMIT"

if [[ -n "$LORA_PATH" ]]; then
    CMD="$CMD --lora_path \"$LORA_PATH\""
fi

echo_info "执行命令: $CMD"
echo ""

# 检查Python和依赖
if ! command -v python &> /dev/null; then
    echo_error "Python未安装或未在PATH中"
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

# 执行命令
echo_info "开始执行..."
if eval $CMD; then
    # 计算执行时间
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))
    
    echo ""
    echo_success "任务完成！"
    echo_info "执行时间: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒"
    echo_info "输出文件: $OUTPUT_FILE"
    
    # 显示输出文件信息
    if [[ -f "$OUTPUT_FILE" ]]; then
        LINE_COUNT=$(wc -l < "$OUTPUT_FILE")
        FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        echo_info "输出统计: $LINE_COUNT 行, $FILE_SIZE"
    fi
else
    echo_error "任务执行失败！"
    exit 1
fi
