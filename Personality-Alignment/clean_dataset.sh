set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/clean_dataset.py"

# Default parameters
INPUT_FILE="dialogue_dataset_all_v6_summarized.jsonl"
OUTPUT_FILE="dialogue_dataset_all_v6_cleaned.jsonl"
STRATEGY="comprehensive"
QWEN_MODEL_PATH="/project/hdtaccuracy/models/base/Qwen3-8B"
DEVICE="auto"
BATCH_SIZE=32
INFERENCE_BATCH_SIZE=32
LOG_FILE=""

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -i, --input FILE          Input jsonl file path (required)"
    echo "  -o, --output FILE         Output jsonl file path (required)"
    echo "  -s, --strategy STRATEGY   Cleaning strategy: typo_only, convert_only, filter, comprehensive (default: comprehensive)"
    echo "  -m, --model PATH          Qwen model path or HuggingFace model name (default: Qwen/Qwen3-8B)"
    echo "  -d, --device DEVICE       Device for inference: auto, cuda, cpu, cuda:0, etc. (default: auto)"
    echo "  -b, --batch-size SIZE     Batch size for processing (default: 32)"
    echo "  --inference-batch-size SIZE  Batch size for model inference (default: 8, adjust based on GPU memory)"
    echo "  -l, --log FILE            Log file path (optional)"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -i data.jsonl -o cleaned_data.jsonl"
    echo "  $0 -i data.jsonl -o cleaned_data.jsonl -s typo_only -d cuda:0"
    echo "  $0 -i data.jsonl -o cleaned_data.jsonl -m /path/to/local/qwen3-8b"
    exit 1
}

# Function to check if file exists
check_file() {
    if [[ ! -f "$1" ]]; then
        echo "Error: File '$1' does not exist!"
        exit 1
    fi
}

# Function to check if directory exists and create if needed
ensure_output_dir() {
    local output_dir=$(dirname "$1")
    if [[ ! -d "$output_dir" ]]; then
        echo "Creating output directory: $output_dir"
        mkdir -p "$output_dir"
    fi
}

# Function to check GPU availability
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Information:"
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
        echo ""
    else
        echo "Warning: nvidia-smi not found. GPU may not be available."
        echo ""
    fi
}

# Function to estimate memory requirements
estimate_memory() {
    local model_path="$1"
    echo "Memory estimation for model: $model_path"
    
    case "$model_path" in
        *"7B"*|*"8B"*)
            echo "  - Model size: ~7-8B parameters"
            echo "  - Estimated GPU memory: 14-16GB (FP16)"
            echo "  - Recommended: RTX 3090/4090, A100, or similar"
            ;;
        *"3B"*|*"4B"*)
            echo "  - Model size: ~3-4B parameters"
            echo "  - Estimated GPU memory: 6-8GB (FP16)"
            echo "  - Recommended: RTX 3080, RTX 4070, or similar"
            ;;
        *"1.5B"*|*"2B"*)
            echo "  - Model size: ~1.5-2B parameters"
            echo "  - Estimated GPU memory: 3-4GB (FP16)"
            echo "  - Recommended: RTX 3060, RTX 4060, or similar"
            ;;
        *)
            echo "  - Model size: Unknown"
            echo "  - Please ensure sufficient GPU memory is available"
            ;;
    esac
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -s|--strategy)
            STRATEGY="$2"
            shift 2
            ;;
        -m|--model)
            QWEN_MODEL_PATH="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --inference-batch-size)
            INFERENCE_BATCH_SIZE="$2"
            shift 2
            ;;
        -l|--log)
            LOG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required parameters
if [[ -z "$INPUT_FILE" ]]; then
    echo "Error: Input file is required!"
    usage
fi

if [[ -z "$OUTPUT_FILE" ]]; then
    echo "Error: Output file is required!"
    usage
fi

# Validate strategy
case "$STRATEGY" in
    typo_only|convert_only|filter|comprehensive)
        ;;
    *)
        echo "Error: Invalid strategy '$STRATEGY'"
        echo "Valid strategies: typo_only, convert_only, filter, comprehensive"
        exit 1
        ;;
esac

# Check input file exists
check_file "$INPUT_FILE"

# Ensure output directory exists
ensure_output_dir "$OUTPUT_FILE"

# Check Python script exists
check_file "$PYTHON_SCRIPT"

# Print configuration
echo "========================================"
echo "Qwen Dataset Cleaning Configuration"
echo "========================================"
echo "Input file:       $INPUT_FILE"
echo "Output file:      $OUTPUT_FILE"
echo "Strategy:         $STRATEGY"
echo "Model path:       $QWEN_MODEL_PATH"
echo "Device:           $DEVICE"
echo "Batch size:       $BATCH_SIZE"
echo "Inference batch:  $INFERENCE_BATCH_SIZE"
echo "Log file:         ${LOG_FILE:-'stdout/stderr'}"
echo "========================================"
echo ""

# Check GPU if using CUDA
if [[ "$DEVICE" == *"cuda"* ]] || [[ "$DEVICE" == "auto" ]]; then
    check_gpu
fi

# Estimate memory requirements
estimate_memory "$QWEN_MODEL_PATH"

# Check if input file has content
line_count=$(wc -l < "$INPUT_FILE")
echo "Input file contains $line_count lines"
echo ""

# Ask for confirmation
read -p "Do you want to proceed with cleaning? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleaning cancelled."
    exit 0
fi

# Prepare Python command
PYTHON_CMD="python $PYTHON_SCRIPT"
PYTHON_CMD="$PYTHON_CMD --cleaner-type qwen"
PYTHON_CMD="$PYTHON_CMD --input '$INPUT_FILE'"
PYTHON_CMD="$PYTHON_CMD --output '$OUTPUT_FILE'"
PYTHON_CMD="$PYTHON_CMD --strategy $STRATEGY"
PYTHON_CMD="$PYTHON_CMD --qwen-model-path '$QWEN_MODEL_PATH'"
PYTHON_CMD="$PYTHON_CMD --device '$DEVICE'"
PYTHON_CMD="$PYTHON_CMD --batch-size $BATCH_SIZE"
PYTHON_CMD="$PYTHON_CMD --inference-batch-size $INFERENCE_BATCH_SIZE"

# Execute cleaning
echo "Starting dataset cleaning..."
echo "Command: $PYTHON_CMD"
echo ""

start_time=$(date +%s)

if [[ -n "$LOG_FILE" ]]; then
    # Run with log file
    ensure_output_dir "$LOG_FILE"
    eval "$PYTHON_CMD" 2>&1 | tee "$LOG_FILE"
    exit_code=${PIPESTATUS[0]}
else
    # Run without log file
    eval "$PYTHON_CMD"
    exit_code=$?
fi

end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "========================================"
if [[ $exit_code -eq 0 ]]; then
    echo "✅ Cleaning completed successfully!"
    echo "⏱️  Total time: ${duration}s"
    
    # Show output file statistics
    if [[ -f "$OUTPUT_FILE" ]]; then
        output_lines=$(wc -l < "$OUTPUT_FILE")
        echo "📊 Output statistics:"
        echo "   - Input lines:  $line_count"
        echo "   - Output lines: $output_lines"
        echo "   - Filtered out: $((line_count - output_lines))"
        echo "   - Success rate: $(( (output_lines * 100) / line_count ))%"
    fi
    
    echo "📁 Output file: $OUTPUT_FILE"
    if [[ -n "$LOG_FILE" ]]; then
        echo "📋 Log file: $LOG_FILE"
    fi
else
    echo "❌ Cleaning failed with exit code: $exit_code"
    echo "⏱️  Time elapsed: ${duration}s"
    
    if [[ -n "$LOG_FILE" ]]; then
        echo "📋 Check log file for details: $LOG_FILE"
    fi
fi
echo "========================================"

exit $exit_code