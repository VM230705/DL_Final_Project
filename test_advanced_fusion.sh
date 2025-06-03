#!/bin/bash

# Test advanced fusion methods - FIXED VERSION
# Usage: ./test_advanced_fusion.sh [dataset] [pred_len] [epochs]

DATASET=${1:-weather}
PRED_LEN=${2:-96}
EPOCHS=${3:-3}
GPU_ID=${4:-0}

echo "🚀 Testing Advanced Fusion Methods (FIXED VERSION)"
echo "Dataset: $DATASET, Prediction Length: $PRED_LEN, Epochs: $EPOCHS"
echo "=============================================="

# Create timestamp for this experiment
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="advanced_fusion_logs_${TIMESTAMP}"
mkdir -p $LOG_DIR

# Test configurations - 只測試新增的advanced fusion方法
declare -a FUSION_METHODS=(
    # "single_scale"
    # "attention"
    "scale_aware_attention"
    "progressive_multires"
)

declare -a MODEL_CONFIGS=(
    "SingleScale_baseline"
    "MultiScale_Attention"
    "MultiScale_ScaleAware"
    "MultiScale_Progressive"
)

# Function to run single experiment
run_experiment() {
    local fusion_method=$1
    local config_name=$2
    local dataset=$3
    local pred_len=$4
    local epochs=$5
    
    echo "🔄 Running: $config_name with $fusion_method fusion"
    
    # Set correct dataset path
    case $dataset in
        "weather")
            DATA_PATH="weather/weather.csv"
            ENC_IN=21
            DEC_IN=21
            C_OUT=21
            ;;
        "ECL")
            DATA_PATH="electricity/electricity.csv"
            ENC_IN=321
            DEC_IN=321
            C_OUT=321
            ;;
        "ETTh1")
            DATA_PATH="ETT-small/ETTh1.csv"
            ENC_IN=7
            DEC_IN=7
            C_OUT=7
            ;;
        *)
            echo "Unknown dataset: $dataset"
            return 1
            ;;
    esac
    
    # Set model parameters based on fusion method
    if [ "$fusion_method" = "single_scale" ]; then
        MULTI_SCALE_ARG=""
        FUSION_TYPE_ARG=""
        PATCH_SIZES_ARG=""
    else
        MULTI_SCALE_ARG="--use_multi_scale"
        FUSION_TYPE_ARG="--fusion_type $fusion_method"
        PATCH_SIZES_ARG="--patch_sizes 8,16,24"
    fi
    
    # Run the experiment
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path $DATA_PATH \
        --model_id "long_term_forecast_${dataset}_96_${pred_len}_TimeXer_${config_name}" \
        --model TimeXer \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 1 \
        --d_layers 1 \
        --factor 3 \
        --enc_in $ENC_IN \
        --dec_in $DEC_IN \
        --c_out $C_OUT \
        --des 'Exp' \
        --itr 1 \
        --train_epochs $epochs \
        --patience 3 \
        --learning_rate 0.01 \
        --patch_len 16 \
        $MULTI_SCALE_ARG \
        $FUSION_TYPE_ARG \
        $PATCH_SIZES_ARG \
        --gpu $GPU_ID \
        2>&1 | tee ${LOG_DIR}/${config_name}_${fusion_method}_${dataset}_${pred_len}.log
    
    # Extract results
    RESULT_FILE="${LOG_DIR}/${config_name}_${fusion_method}_${dataset}_${pred_len}.log"
    if [ -f "$RESULT_FILE" ]; then
        echo "📊 Results for $config_name:" >> ${LOG_DIR}/summary.txt
        grep -E "mse:|mae:" $RESULT_FILE | tail -1 >> ${LOG_DIR}/summary.txt
        echo "---" >> ${LOG_DIR}/summary.txt
    fi
    
    echo "✅ Completed: $config_name"
    echo ""
}

# Main execution
echo "🎯 Starting Advanced Fusion Experiments (Fixed Version)"
echo "Results will be saved in: $LOG_DIR"
echo ""

# Run all experiments
for i in "${!FUSION_METHODS[@]}"; do
    fusion_method="${FUSION_METHODS[$i]}"
    config_name="${MODEL_CONFIGS[$i]}"
    
    run_experiment "$fusion_method" "$config_name" "$DATASET" "$PRED_LEN" "$EPOCHS"
done

echo "🏁 All experiments completed!"
echo "Check results in: $LOG_DIR/summary.txt"

# Generate quick comparison
echo "📈 Quick Results Comparison:"
if [ -f "${LOG_DIR}/summary.txt" ]; then
    cat ${LOG_DIR}/summary.txt
fi

echo ""
echo "🎉 Advanced Fusion Testing Complete!"
echo "📁 All logs and results saved in: $LOG_DIR"