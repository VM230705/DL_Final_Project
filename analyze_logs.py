#!/usr/bin/env python3
import os
import re
import pandas as pd
import glob
from pathlib import Path

def extract_info_from_log(log_file_path):
    """從log檔案中提取關鍵資訊"""
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取基本資訊
        dataset = None
        pred_len = None
        seq_len = None
        fusion_type = "SingleScale"  # 默認值
        mse = None
        mae = None
        
        # 從檔案名或內容提取資訊
        filename = os.path.basename(log_file_path)
        
        # 提取資料集名稱
        dataset_patterns = [
            r'ETTh1', r'ETTh2', r'ETTm1', r'ETTm2', 
            r'ECL', r'Exchange', r'Weather', r'weather', 
            r'Traffic', r'traffic'
        ]
        for pattern in dataset_patterns:
            if re.search(pattern, content, re.IGNORECASE) or re.search(pattern, filename, re.IGNORECASE):
                dataset = pattern
                break
        
        # 從Args中提取預測長度
        pred_len_match = re.search(r'Pred Len:\s+(\d+)', content)
        if pred_len_match:
            pred_len = int(pred_len_match.group(1))
        
        # 從Args中提取序列長度
        seq_len_match = re.search(r'Seq Len:\s+(\d+)', content)
        if seq_len_match:
            seq_len = int(seq_len_match.group(1))
        
        # 提取fusion類型
        if 'MultiScale' in content or 'MultiScale' in filename:
            fusion_patterns = [
                'ScaleAwareAttention', 'ScaleAware', 'Attention',  
                'Gated', 'Hierarchical', 'Concat', 'ProgressiveMultires'
            ]
            for pattern in fusion_patterns:
                if pattern in content or pattern in filename:
                    fusion_type = f"MultiScale_{pattern}"
                    break
            if fusion_type == "SingleScale":  # 如果沒找到具體類型，但有MultiScale
                fusion_type = "MultiScale"
        
        # 提取最終的MSE和MAE結果
        mse_mae_match = re.search(r'mse:([\d.]+),\s*mae:([\d.]+)', content)
        if mse_mae_match:
            mse = float(mse_mae_match.group(1))
            mae = float(mse_mae_match.group(2))
        
        # 提取模型參數
        d_model_match = re.search(r'd model:\s+(\d+)', content)
        d_model = int(d_model_match.group(1)) if d_model_match else None
        
        n_heads_match = re.search(r'n heads:\s+(\d+)', content)
        n_heads = int(n_heads_match.group(1)) if n_heads_match else None
        
        e_layers_match = re.search(r'e layers:\s+(\d+)', content)
        e_layers = int(e_layers_match.group(1)) if e_layers_match else None
        
        d_ff_match = re.search(r'd FF:\s+(\d+)', content)
        d_ff = int(d_ff_match.group(1)) if d_ff_match else None
        
        # 如果從檔案名無法提取資訊，嘗試從內容的實驗名稱提取
        if not dataset or not pred_len:
            exp_match = re.search(r'long_term_forecast_([^_]+)_(\d+)_(\d+)', content)
            if exp_match:
                if not dataset:
                    dataset = exp_match.group(1)
                if not pred_len:
                    pred_len = int(exp_match.group(3))
                if not seq_len:
                    seq_len = int(exp_match.group(2))
        
        return {
            'log_file': log_file_path,
            'dataset': dataset,
            'seq_len': seq_len,
            'pred_len': pred_len,
            'fusion_type': fusion_type,
            'mse': mse,
            'mae': mae,
            'd_model': d_model,
            'n_heads': n_heads,
            'e_layers': e_layers,
            'd_ff': d_ff
        }
        
    except Exception as e:
        print(f"Error processing {log_file_path}: {e}")
        return None

def process_log_directory(directory_path, output_csv="log_analysis_results_grouped.csv"):
    """處理指定目錄下的所有log檔案"""
    
    # 尋找所有log檔案
    log_files = []
    
    # 支援多種路徑格式
    if os.path.isdir(directory_path):
        # 遞迴搜尋所有.log檔案
        log_files = glob.glob(os.path.join(directory_path, "**/*.log"), recursive=True)
    else:
        print(f"Directory {directory_path} does not exist")
        return
    
    print(f"Found {len(log_files)} log files")
    
    # 處理所有log檔案
    results = []
    for log_file in log_files:
        print(f"Processing: {log_file}")
        result = extract_info_from_log(log_file)
        if result and result['mse'] is not None:
            results.append(result)
    
    if not results:
        print("No valid results found")
        return
    
    # 轉換為DataFrame
    df = pd.DataFrame(results)
    
    # 過濾掉缺少關鍵資訊的記錄
    df = df.dropna(subset=['dataset', 'pred_len', 'mse', 'mae'])
    
    # 創建dataset_pred_len組合列用於分組
    df['dataset_pred_len'] = df['dataset'].astype(str) + '_' + df['pred_len'].astype(str)
    
    # 按照dataset, pred_len分組，然後在每組內按MSE從高到低排序
    df_sorted = df.sort_values(['dataset', 'pred_len', 'mse'], ascending=[True, True, True])
    
    # 選擇要輸出的列
    output_columns = [
        'dataset', 'pred_len', 'fusion_type', 'mse', 'mae', 
        'd_model', 'n_heads', 'e_layers', 'd_ff', 'seq_len'
    ]
    
    # 只保留存在的列
    available_columns = [col for col in output_columns if col in df_sorted.columns]
    final_df = df_sorted[available_columns]
    
    # 保存到CSV
    final_df.to_csv(output_csv, index=False, float_format='%.6f')
    print(f"\nResults saved to {output_csv}")
    
    # 顯示統計資訊
    print(f"\nSummary:")
    print(f"Total experiments: {len(df)}")
    print(f"Unique dataset-pred_len combinations: {len(df['dataset_pred_len'].unique())}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    print(f"Prediction lengths: {sorted(df['pred_len'].unique())}")
    print(f"Fusion types: {sorted(df['fusion_type'].unique())}")
    
    # 顯示按組別的結果
    print(f"\nResults by dataset-pred_len groups (MSE from high to low within each group):")
    print("=" * 80)
    
    for group_name, group_data in df_sorted.groupby(['dataset', 'pred_len']):
        dataset, pred_len = group_name
        print(f"\n{dataset} - {pred_len}:")
        print("-" * 40)
        group_display = group_data[['fusion_type', 'mse', 'mae']].copy()
        group_display.index = range(1, len(group_display) + 1)  # 重新編號從1開始
        print(group_display.to_string())
    
    return final_df

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze log files and generate grouped CSV report')
    parser.add_argument('--dir', '-d', required=True, help='Directory containing log files')
    parser.add_argument('--output', '-o', default='log_analysis_results_grouped.csv', help='Output CSV file name')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        print(f"Error: Directory '{args.dir}' does not exist")
        return
    
    print(f"Analyzing log files in: {args.dir}")
    print(f"Output file: {args.output}")
    print("-" * 50)
    
    process_log_directory(args.dir, args.output)

if __name__ == "__main__":
    # 如果直接運行，使用當前目錄
    if len(os.sys.argv) == 1:
        # 自動搜尋當前目錄下的log檔案
        # fusion_comparison_logs_20250603_071201_ETTh1
        # fusion_comparison_logs_20250603_071303_ETTh2
        # fusion_comparison_logs_20250603_145950_ETTm1
        # fusion_comparison_logs_20250603_150107_ETTm2
        # fusion_comparison_logs_20250602_210417_weather
        # fusion_comparison_logs_20250603_093127_weather_sec
        # fusion_comparison_logs_20250603_060610_ECL
        # fusion_comparison_logs_20250603_172323_traffic
        current_dir = "./fusion_comparison_logs_20250603_172323_traffic"
        print("No directory specified, using current directory")
        process_log_directory(current_dir, "log_analysis_results_grouped.csv")
    else:
        main()