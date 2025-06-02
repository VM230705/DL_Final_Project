#!/bin/bash

# =============================================================================
# Multi-Scale Fusion Methods Comparison Script
# Based on Weather Dataset TimeXer.sh original hyperparameters
# =============================================================================

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
LOG_DIR="./fusion_comparison_logs_${TIMESTAMP}"
RESULT_FILE="./fusion_comparison_results_${TIMESTAMP}.txt"

# Default parameters (can be overridden by command line arguments)
DATASET="weather"
PRED_LENS="96,192,336,720"
FUSION_METHODS="single_scale,multi_attention,multi_gated,multi_hierarchical,multi_concat"
EPOCHS=10
# Check if CUDA_VISIBLE_DEVICES is set, use it as default GPU_ID
if [[ -n "$CUDA_VISIBLE_DEVICES" && "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]; then
    GPU_ID=$CUDA_VISIBLE_DEVICES
else
    GPU_ID=7
fi

# Dataset configurations based on original scripts
declare -A DATASET_CONFIGS=(
    ["weather_root_path"]="./dataset/weather/"
    ["weather_data_path"]="weather.csv"
    ["weather_data"]="custom"
    ["weather_enc_in"]="21"
    ["weather_dec_in"]="21"
    ["weather_c_out"]="21"
    
    ["ETTh1_root_path"]="./dataset/ETT-small/"
    ["ETTh1_data_path"]="ETTh1.csv"
    ["ETTh1_data"]="ETTh1"
    ["ETTh1_enc_in"]="7"
    ["ETTh1_dec_in"]="7"
    ["ETTh1_c_out"]="7"
    
    ["ETTh2_root_path"]="./dataset/ETT-small/"
    ["ETTh2_data_path"]="ETTh2.csv"
    ["ETTh2_data"]="ETTh2"
    ["ETTh2_enc_in"]="7"
    ["ETTh2_dec_in"]="7"
    ["ETTh2_c_out"]="7"
    
    ["ETTm1_root_path"]="./dataset/ETT-small/"
    ["ETTm1_data_path"]="ETTm1.csv"
    ["ETTm1_data"]="ETTm1"
    ["ETTm1_enc_in"]="7"
    ["ETTm1_dec_in"]="7"
    ["ETTm1_c_out"]="7"
    
    ["ETTm2_root_path"]="./dataset/ETT-small/"
    ["ETTm2_data_path"]="ETTm2.csv"
    ["ETTm2_data"]="ETTm2"
    ["ETTm2_enc_in"]="7"
    ["ETTm2_dec_in"]="7"
    ["ETTm2_c_out"]="7"
    
    ["ECL_root_path"]="./dataset/electricity/"
    ["ECL_data_path"]="electricity.csv"
    ["ECL_data"]="custom"
    ["ECL_enc_in"]="321"
    ["ECL_dec_in"]="321"
    ["ECL_c_out"]="321"
    
    ["traffic_root_path"]="./dataset/traffic/"
    ["traffic_data_path"]="traffic.csv"
    ["traffic_data"]="custom"
    ["traffic_enc_in"]="862"
    ["traffic_dec_in"]="862"
    ["traffic_c_out"]="862"
)

# Weather dataset prediction length specific configurations (from original script)
declare -A WEATHER_CONFIGS=(
    ["96_d_model"]="256"
    ["96_d_ff"]="512"
    ["96_e_layers"]="1"
    ["96_batch_size"]="4"
    
    ["192_d_model"]="128"
    ["192_d_ff"]="1024"
    ["192_e_layers"]="3"
    ["192_batch_size"]="4"
    
    ["336_d_model"]="256"
    ["336_d_ff"]="2048"
    ["336_e_layers"]="1"
    ["336_batch_size"]="4"
    
    ["720_d_model"]="128"
    ["720_d_ff"]="2048"
    ["720_e_layers"]="1"
    ["720_batch_size"]="4"
)

# ETTh1 configurations
declare -A ETTH1_CONFIGS=(
    ["96_d_model"]="128"
    ["96_d_ff"]="256"
    ["96_e_layers"]="2"
    ["96_batch_size"]="4"
    
    ["192_d_model"]="128"
    ["192_d_ff"]="2048"
    ["192_e_layers"]="2"
    ["192_batch_size"]="4"
    
    ["336_d_model"]="512"
    ["336_d_ff"]="1024"
    ["336_e_layers"]="1"
    ["336_batch_size"]="16"
    
    ["720_d_model"]="256"
    ["720_d_ff"]="1024"
    ["720_e_layers"]="1"
    ["720_batch_size"]="16"
)

# ETTh2 configurations
declare -A ETTH2_CONFIGS=(
    ["96_d_model"]="256"
    ["96_d_ff"]="1024"
    ["96_e_layers"]="1"
    ["96_batch_size"]="4"
    
    ["192_d_model"]="256"
    ["192_d_ff"]="1024"
    ["192_e_layers"]="1"
    ["192_batch_size"]="4"
    
    ["336_d_model"]="512"
    ["336_d_ff"]="1024"
    ["336_e_layers"]="2"
    ["336_batch_size"]="4"
    
    ["720_d_model"]="256"
    ["720_d_ff"]="1024"
    ["720_e_layers"]="2"
    ["720_batch_size"]="16"
)

# ETTm1 configurations
declare -A ETTM1_CONFIGS=(
    ["96_d_model"]="256"
    ["96_d_ff"]="2048"
    ["96_e_layers"]="1"
    ["96_batch_size"]="4"
    
    ["192_d_model"]="256"
    ["192_d_ff"]="256"
    ["192_e_layers"]="1"
    ["192_batch_size"]="4"
    
    ["336_d_model"]="256"
    ["336_d_ff"]="1024"
    ["336_e_layers"]="1"
    ["336_batch_size"]="4"
    
    ["720_d_model"]="256"
    ["720_d_ff"]="512"
    ["720_e_layers"]="1"
    ["720_batch_size"]="4"
)

# ETTm2 configurations
declare -A ETTM2_CONFIGS=(
    ["96_d_model"]="256"
    ["96_d_ff"]="2048"
    ["96_e_layers"]="1"
    ["96_batch_size"]="4"
    
    ["192_d_model"]="256"
    ["192_d_ff"]="1024"
    ["192_e_layers"]="1"
    ["192_batch_size"]="4"
    
    ["336_d_model"]="512"
    ["336_d_ff"]="1024"
    ["336_e_layers"]="1"
    ["336_batch_size"]="4"
    
    ["720_d_model"]="512"
    ["720_d_ff"]="2048"
    ["720_e_layers"]="1"
    ["720_batch_size"]="4"
)

# ECL configurations
declare -A ECL_CONFIGS=(
    ["96_d_model"]="512"
    ["96_d_ff"]="512"
    ["96_e_layers"]="4"
    ["96_batch_size"]="4"
    
    ["192_d_model"]="512"
    ["192_d_ff"]="2048"
    ["192_e_layers"]="3"
    ["192_batch_size"]="4"
    
    ["336_d_model"]="512"
    ["336_d_ff"]="2048"
    ["336_e_layers"]="4"
    ["336_batch_size"]="4"
    
    ["720_d_model"]="512"
    ["720_d_ff"]="2048"
    ["720_e_layers"]="3"
    ["720_batch_size"]="4"
)

# Traffic configurations
declare -A TRAFFIC_CONFIGS=(
    ["96_d_model"]="512"
    ["96_d_ff"]="512"
    ["96_e_layers"]="3"
    ["96_batch_size"]="4"
    
    ["192_d_model"]="512"
    ["192_d_ff"]="512"
    ["192_e_layers"]="3"
    ["192_batch_size"]="4"
    
    ["336_d_model"]="512"
    ["336_d_ff"]="512"
    ["336_e_layers"]="2"
    ["336_batch_size"]="4"
    
    ["720_d_model"]="512"
    ["720_d_ff"]="512"
    ["720_e_layers"]="2"
    ["720_batch_size"]="4"
)

# Function to print usage
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  -d, --dataset DATASET        Dataset to use (weather,ETTh1,ETTh2,ETTm1,ETTm2,ECL,traffic) [default: weather]"
    echo "  -p, --pred_lens LENGTHS      Comma-separated prediction lengths [default: 96,192,336,720]"
    echo "  -f, --fusion_methods METHODS Comma-separated fusion methods [default: single_scale,multi_attention,multi_gated,multi_hierarchical,multi_concat]"
    echo "  -e, --epochs EPOCHS          Number of training epochs [default: 10]"
    echo "  -g, --gpu GPU_ID             GPU ID to use [default: 0]"
    echo "  -h, --help                   Show this help message"
    echo ""
    echo "Available fusion methods:"
    echo "  - single_scale: Original TimeXer with single patch size (16)"
    echo "  - multi_attention: Multi-scale with attention fusion"
    echo "  - multi_gated: Multi-scale with gated fusion"
    echo "  - multi_hierarchical: Multi-scale with hierarchical fusion"
    echo "  - multi_concat: Multi-scale with concatenation fusion"
    echo ""
    echo "Examples:"
    echo "  $0 --dataset weather --pred_lens 96,192 --fusion_methods single_scale,multi_attention"
    echo "  $0 -d ETTh1 -p 96,336,720 -f multi_attention,multi_gated -e 5"
}

# Function to get dataset configuration
get_dataset_config() {
    local dataset=$1
    local key=$2
    local config_key="${dataset}_${key}"
    echo "${DATASET_CONFIGS[$config_key]}"
}

# Function to get prediction length specific configuration
get_pred_config() {
    local dataset=$1
    local pred_len=$2
    local key=$3
    
    local config_var="${dataset^^}_CONFIGS"  # Convert to uppercase
    local config_key="${pred_len}_${key}"
    
    case $dataset in
        "weather")
            echo "${WEATHER_CONFIGS[$config_key]}"
            ;;
        "ETTh1")
            echo "${ETTH1_CONFIGS[$config_key]}"
            ;;
        "ETTh2")
            echo "${ETTH2_CONFIGS[$config_key]}"
            ;;
        "ETTm1")
            echo "${ETTM1_CONFIGS[$config_key]}"
            ;;
        "ETTm2")
            echo "${ETTM2_CONFIGS[$config_key]}"
            ;;
        "ECL")
            echo "${ECL_CONFIGS[$config_key]}"
            ;;
        "traffic")
            echo "${TRAFFIC_CONFIGS[$config_key]}"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Function to run single experiment
run_experiment() {
    local dataset=$1
    local pred_len=$2
    local fusion_method=$3
    local log_file=$4
    
    echo "🚀 Running: $fusion_method on $dataset (pred_len=$pred_len)"
    
    # Get dataset configuration
    local root_path=$(get_dataset_config $dataset "root_path")
    local data_path=$(get_dataset_config $dataset "data_path")
    local data_type=$(get_dataset_config $dataset "data")
    local enc_in=$(get_dataset_config $dataset "enc_in")
    local dec_in=$(get_dataset_config $dataset "dec_in")
    local c_out=$(get_dataset_config $dataset "c_out")
    
    # Get prediction length specific configuration
    local d_model=$(get_pred_config $dataset $pred_len "d_model")
    local d_ff=$(get_pred_config $dataset $pred_len "d_ff")
    local e_layers=$(get_pred_config $dataset $pred_len "e_layers")
    local batch_size=$(get_pred_config $dataset $pred_len "batch_size")
    
    # Check if configuration exists
    if [[ -z "$d_model" || -z "$d_ff" || -z "$e_layers" || -z "$batch_size" ]]; then
        echo "❌ No configuration found for $dataset pred_len=$pred_len"
        return 1
    fi
    
    # Build base command
    local cmd="python3 -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path $root_path \
        --data_path $data_path \
        --model_id ${dataset}_96_${pred_len} \
        --model TimeXer \
        --data $data_type \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --enc_in $enc_in \
        --dec_in $dec_in \
        --c_out $c_out \
        --d_model $d_model \
        --d_ff $d_ff \
        --e_layers $e_layers \
        --batch_size $batch_size \
        --n_heads 8 \
        --d_layers 1 \
        --factor 3 \
        --dropout 0.1 \
        --embed timeF \
        --freq h \
        --train_epochs $EPOCHS \
        --patience 3 \
        --learning_rate 0.0001 \
        --itr 1"
    
    # Add fusion-specific parameters
    case $fusion_method in
        "single_scale")
            cmd="$cmd --des SingleScale_${dataset}"
            ;;
        "multi_attention")
            cmd="$cmd --use_multi_scale --fusion_type attention --patch_sizes 8,16,24 --des MultiScale_${dataset}_Attention"
            ;;
        "multi_gated")
            cmd="$cmd --use_multi_scale --fusion_type gated --patch_sizes 8,16,24 --des MultiScale_${dataset}_Gated"
            ;;
        "multi_hierarchical")
            cmd="$cmd --use_multi_scale --fusion_type hierarchical --patch_sizes 8,16,24 --des MultiScale_${dataset}_Hierarchical"
            ;;
        "multi_concat")
            cmd="$cmd --use_multi_scale --fusion_type concat --patch_sizes 8,16,24 --des MultiScale_${dataset}_Concat"
            ;;
        *)
            echo "❌ Unknown fusion method: $fusion_method"
            return 1
            ;;
    esac
    
    # Run experiment
    echo "   Command: $(echo $cmd | cut -d' ' -f1-8)..."
    local start_time=$(date +%s)
    
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    if eval $cmd > "$log_file" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "   ✅ Success! Duration: ${duration}s"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "   ❌ Failed! Duration: ${duration}s"
        echo "   Check log: $log_file"
        return 1
    fi
}

# Function to parse results from log
parse_results() {
    local log_file=$1
    local mse=$(grep -o "mse:[0-9]*\.[0-9]*" "$log_file" | tail -1 | cut -d: -f2)
    local mae=$(grep -o "mae:[0-9]*\.[0-9]*" "$log_file" | tail -1 | cut -d: -f2)
    echo "$mse,$mae"
}

# Function to generate summary report
generate_report() {
    local dataset=$1
    local pred_lens_array=($2)
    local fusion_methods_array=($3)
    
    echo "# Multi-Scale Fusion Methods Comparison Report" > "$RESULT_FILE"
    echo "Generated: $(date)" >> "$RESULT_FILE"
    echo "Dataset: $dataset" >> "$RESULT_FILE"
    echo "Prediction lengths: ${pred_lens_array[*]}" >> "$RESULT_FILE"
    echo "Fusion methods: ${fusion_methods_array[*]}" >> "$RESULT_FILE"
    echo "" >> "$RESULT_FILE"
    
    echo "## Results Table" >> "$RESULT_FILE"
    echo "" >> "$RESULT_FILE"
    echo "| Pred Length | Method | MSE | MAE | Status |" >> "$RESULT_FILE"
    echo "|-------------|--------|-----|-----|--------|" >> "$RESULT_FILE"
    
    for pred_len in "${pred_lens_array[@]}"; do
        for method in "${fusion_methods_array[@]}"; do
            local log_file="$LOG_DIR/${method}_${dataset}_${pred_len}.log"
            if [[ -f "$log_file" ]]; then
                local results=$(parse_results "$log_file")
                local mse=$(echo $results | cut -d, -f1)
                local mae=$(echo $results | cut -d, -f2)
                local status="✅"
                
                if [[ -z "$mse" || -z "$mae" ]]; then
                    mse="N/A"
                    mae="N/A"
                    status="❌"
                fi
                
                echo "| $pred_len | $method | $mse | $mae | $status |" >> "$RESULT_FILE"
            else
                echo "| $pred_len | $method | N/A | N/A | ❌ |" >> "$RESULT_FILE"
            fi
        done
    done
    
    echo "" >> "$RESULT_FILE"
    echo "Logs directory: $LOG_DIR" >> "$RESULT_FILE"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -p|--pred_lens)
            PRED_LENS="$2"
            shift 2
            ;;
        -f|--fusion_methods)
            FUSION_METHODS="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Convert comma-separated strings to arrays
IFS=',' read -ra PRED_LENS_ARRAY <<< "$PRED_LENS"
IFS=',' read -ra FUSION_METHODS_ARRAY <<< "$FUSION_METHODS"

# Validate dataset
if [[ ! " weather ETTh1 ETTh2 ETTm1 ETTm2 ECL traffic " =~ " $DATASET " ]]; then
    echo "❌ Invalid dataset: $DATASET"
    echo "Available datasets: weather, ETTh1, ETTh2, ETTm1, ETTm2, ECL, traffic"
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"

# Print configuration
echo "🔬 MULTI-SCALE FUSION METHOD COMPARISON"
echo "========================================"
echo "Dataset: $DATASET"
echo "Prediction lengths: ${PRED_LENS_ARRAY[*]}"
echo "Fusion methods: ${FUSION_METHODS_ARRAY[*]}"
echo "Epochs: $EPOCHS"
echo "GPU ID: $GPU_ID"
echo "Log directory: $LOG_DIR"
echo "========================================"

# Run experiments
total_experiments=$((${#PRED_LENS_ARRAY[@]} * ${#FUSION_METHODS_ARRAY[@]}))
current_exp=0

for pred_len in "${PRED_LENS_ARRAY[@]}"; do
    for method in "${FUSION_METHODS_ARRAY[@]}"; do
        current_exp=$((current_exp + 1))
        echo ""
        echo "[$current_exp/$total_experiments] Running experiment..."
        
        log_file="$LOG_DIR/${method}_${DATASET}_${pred_len}.log"
        run_experiment "$DATASET" "$pred_len" "$method" "$log_file"
    done
done

# Generate summary report
echo ""
echo "📊 Generating summary report..."
generate_report "$DATASET" "${PRED_LENS_ARRAY[*]}" "${FUSION_METHODS_ARRAY[*]}"

echo ""
echo "🎉 COMPARISON COMPLETED!"
echo "Results saved to: $RESULT_FILE"
echo "Logs directory: $LOG_DIR"