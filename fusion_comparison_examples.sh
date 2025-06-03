#!/bin/bash

# =============================================================================
# Example usage scripts for compare_fusion_methods.sh
# =============================================================================

echo "🔬 Multi-Scale Fusion Methods Comparison Examples"
echo "================================================="

# Example 1: Quick test with Weather dataset (default), only two methods
echo ""
echo "Example 1: Quick comparison on Weather dataset"
echo "Command: ./compare_fusion_methods.sh -d weather -p 96 -f single_scale,multi_attention -e 2"
echo ""

# Example 2: Compare all methods on ETTh1 for specific prediction lengths
echo "Example 2: Full comparison on ETTh1 dataset"
echo "Command: ./compare_fusion_methods.sh --dataset ETTh1 --pred_lens 96,336 --fusion_methods single_scale,multi_attention,multi_gated,multi_hierarchical,multi_concat --epochs 5"
echo ""

# Example 3: Test specific fusion methods on Traffic dataset
echo "Example 3: Test specific methods on Traffic dataset"
echo "Command: ./compare_fusion_methods.sh -d traffic -p 96,192 -f multi_attention,multi_gated -e 3 -g 0"
echo ""

# Example 4: Full benchmark on multiple datasets
echo "Example 4: Run full benchmark (Warning: This will take a long time!)"
echo "For each dataset, run:"
echo "  ./compare_fusion_methods.sh -d weather -e 10"
echo "  ./compare_fusion_methods.sh -d ETTh1 -e 10"
echo "  ./compare_fusion_methods.sh -d ETTh2 -e 10"
echo "  ./compare_fusion_methods.sh -d ETTm1 -e 10"
echo "  ./compare_fusion_methods.sh -d ETTm2 -e 10"
echo "  ./compare_fusion_methods.sh -d ECL -e 10"
echo "  ./compare_fusion_methods.sh -d traffic -e 10"
echo ""

echo "📋 Available Options:"
echo "  Datasets: weather, ETTh1, ETTh2, ETTm1, ETTm2, ECL, traffic"
echo "  Prediction lengths: 96, 192, 336, 720"
echo "  Fusion methods: single_scale, multi_attention, multi_gated, multi_hierarchical, multi_concat"
echo ""

echo "💡 Tips:"
echo "  - Start with small epochs (-e 2) for testing"
echo "  - Use single prediction length (-p 96) for quick tests"
echo "  - Check GPU availability with nvidia-smi before running"
echo "  - Results will be saved in timestamped directories"
echo ""

echo "📖 For help: ./compare_fusion_methods.sh --help"