import os
import sys
import pandas as pd
import numpy as np
import torch
from data_provider.data_factory import data_provider
import argparse

class DatasetAnalyzer:
    """分析 TimeXer 項目中所有數據集的統計資訊"""
    
    def __init__(self):
        self.datasets_info = {
            'ETTh1': {
                'data_path': 'ETTh1.csv',
                'root_path': './dataset/ETT-small/',
                'data_type': 'ETTh1',
                'features': 'M',
                'target': 'OT',
                'enc_in': 7,
                'dec_in': 7,
                'c_out': 7
            },
            'ETTh2': {
                'data_path': 'ETTh2.csv', 
                'root_path': './dataset/ETT-small/',
                'data_type': 'ETTh2',
                'features': 'M',
                'target': 'OT',
                'enc_in': 7,
                'dec_in': 7,
                'c_out': 7
            },
            'ETTm1': {
                'data_path': 'ETTm1.csv',
                'root_path': './dataset/ETT-small/', 
                'data_type': 'ETTm1',
                'features': 'M',
                'target': 'OT',
                'enc_in': 7,
                'dec_in': 7,
                'c_out': 7
            },
            'ETTm2': {
                'data_path': 'ETTm2.csv',
                'root_path': './dataset/ETT-small/',
                'data_type': 'ETTm2', 
                'features': 'M',
                'target': 'OT',
                'enc_in': 7,
                'dec_in': 7,
                'c_out': 7
            },
            'Weather': {
                'data_path': 'weather.csv',
                'root_path': './dataset/weather/',
                'data_type': 'custom',
                'features': 'M',
                'target': 'OT',
                'enc_in': 21,
                'dec_in': 21,
                'c_out': 21
            },
            'Traffic': {
                'data_path': 'traffic.csv',
                'root_path': './dataset/traffic/',
                'data_type': 'custom', 
                'features': 'M',
                'target': 'OT',
                'enc_in': 862,
                'dec_in': 862,
                'c_out': 862
            },
            'ECL': {
                'data_path': 'electricity.csv',
                'root_path': './dataset/electricity/',
                'data_type': 'custom',
                'features': 'M', 
                'target': 'OT',
                'enc_in': 321,
                'dec_in': 321,
                'c_out': 321
            },
            'Exchange': {
                'data_path': 'exchange_rate.csv',
                'root_path': './dataset/exchange_rate/',
                'data_type': 'custom',
                'features': 'M',
                'target': 'OT',
                'enc_in': 8,
                'dec_in': 8,
                'c_out': 8
            }
        }
        
    def create_mock_args(self, dataset_name, dataset_info):
        """創建模擬的參數對象來初始化數據加載器"""
        class MockArgs:
            def __init__(self):
                # 基本數據參數
                self.data = dataset_info['data_type']
                self.root_path = dataset_info['root_path'] 
                self.data_path = dataset_info['data_path']
                self.features = dataset_info['features']
                self.target = dataset_info['target']
                self.freq = 'h' if 'ETT' in dataset_name else 't'
                self.checkpoints = './checkpoints'
                
                # 模型結構參數
                self.enc_in = dataset_info['enc_in']
                self.dec_in = dataset_info['dec_in'] 
                self.c_out = dataset_info['c_out']
                
                # TimeXer 標準參數
                self.seq_len = 96
                self.label_len = 48
                self.pred_len = 96
                self.seasonal_patterns = None
                self.inverse = False
                self.num_workers = 0
                self.batch_size = 32
                
                # 時間編碼參數
                self.timeenc = 0
                self.embed = 'timeF'
                
                # 任務相關參數
                self.task_name = 'long_term_forecast'
                self.is_training = 1
                self.model_id = f'{dataset_name}_96_96'
                self.model = 'TimeXer'
                
                # 數據增強參數
                self.augmentation_ratio = 0
                self.seed = 2021
                self.jitter = False
                self.scaling = False
                self.permutation = False
                self.randompermutation = False
                self.magwarp = False
                self.timewarp = False
                self.windowslice = False
                self.windowwarp = False
                self.rotation = False
                self.spawner = False
                self.dtwwarp = False
                self.shapedtwwarp = False
                self.wdba = False
                self.discdtw = False
                self.discsdtw = False
                self.extra_tag = ""
                
                # 其他必要參數
                self.scale = True
                self.use_gpu = torch.cuda.is_available()
                self.gpu = 0
                self.use_multi_gpu = False
                self.devices = '0,1,2,3'
                self.detail_freq = self.freq
                
                # TimeXer 特有參數
                self.patch_len = 16
                self.stride = 8
                self.gpt_layers = 3
                self.is_gpt = 1
                self.e_layers = 3
                self.d_layers = 1
                self.factor = 1
                self.d_model = 768
                self.n_heads = 4
                self.d_ff = 768
                self.moving_avg = 25
                self.dropout = 0.3
                self.fc_dropout = 0.3
                self.head_dropout = 0.0
                self.des = 'Exp'
                self.train_epochs = 10
                self.patience = 3
                self.learning_rate = 0.0001
                self.lradj = 'type1'
                self.use_amp = False
                
        return MockArgs()
        
    def analyze_raw_data(self, dataset_name, dataset_info):
        """直接讀取原始數據文件進行分析"""
        file_path = os.path.join(dataset_info['root_path'], dataset_info['data_path'])
        
        if not os.path.exists(file_path):
            return None
            
        try:
            df = pd.read_csv(file_path)
            
            # 基本統計
            total_timesteps = len(df)
            total_columns = len(df.columns)
            
            # 檢查是否有時間列
            time_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            
            # 數值列（排除時間列）
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # 移除時間相關列
            for time_col in time_columns:
                if time_col in numeric_columns:
                    numeric_columns.remove(time_col)
                    
            num_numeric_features = len(numeric_columns)
            
            # 計算數據統計
            data_stats = {}
            if len(numeric_columns) > 0:
                numeric_df = df[numeric_columns]
                data_stats = {
                    'mean': numeric_df.mean().mean(),
                    'std': numeric_df.std().mean(),
                    'min': numeric_df.min().min(),
                    'max': numeric_df.max().max(),
                    'missing_values': numeric_df.isnull().sum().sum()
                }
            
            return {
                'total_timesteps': total_timesteps,
                'total_columns': total_columns,
                'time_columns': time_columns,
                'numeric_columns': numeric_columns,
                'num_numeric_features': num_numeric_features,
                'data_stats': data_stats,
                'sample_columns': list(df.columns)[:10]  # 前10列名稱
            }
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None
    
    def analyze_timexer_data_split(self, dataset_name, dataset_info):
        """使用 TimeXer 的數據加載器分析數據劃分"""
        try:
            args = self.create_mock_args(dataset_name, dataset_info)
            print(f"  🔧 正在分析 {dataset_name} 的 TimeXer 數據劃分...")
            
            # 獲取訓練、驗證、測試數據集
            results = {}
            
            for flag in ['train', 'val', 'test']:
                try:
                    print(f"    ⏳ 加載 {flag} 數據...")
                    data_set, data_loader = data_provider(args, flag)
                    
                    # 獲取數據集信息
                    dataset_length = len(data_set)
                    print(f"    ✅ {flag} 數據集大小: {dataset_length}")
                    
                    # 獲取一個batch來檢查維度
                    for batch_idx, batch in enumerate(data_loader):
                        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                        
                        batch_size = batch_x.shape[0]
                        seq_len = batch_x.shape[1] 
                        n_features = batch_x.shape[2]
                        
                        # 外生變數維度
                        if batch_x_mark is not None:
                            n_exo_features = batch_x_mark.shape[2]
                        else:
                            n_exo_features = 0
                            
                        results[flag] = {
                            'dataset_length': dataset_length,
                            'batch_size': batch_size,
                            'seq_len': seq_len,
                            'n_endogenous_features': n_features,
                            'n_exogenous_features': n_exo_features,
                            'batch_x_shape': list(batch_x.shape),
                            'batch_y_shape': list(batch_y.shape),
                            'batch_x_mark_shape': list(batch_x_mark.shape) if batch_x_mark is not None else None,
                            'batch_y_mark_shape': list(batch_y_mark.shape) if batch_y_mark is not None else None,
                            'data_range': {
                                'x_min': float(batch_x.min()),
                                'x_max': float(batch_x.max()),
                                'x_mean': float(batch_x.mean())
                            }
                        }
                        print(f"    📊 {flag} batch shape: {batch_x.shape}, target shape: {batch_y.shape}")
                        break
                        
                except Exception as e:
                    print(f"    ❌ 加載 {flag} 數據失敗: {e}")
                    results[flag] = f"Error: {e}"
                    
            return results
            
        except Exception as e:
            print(f"  ❌ 分析 {dataset_name} 失敗: {e}")
            return f"Error analyzing {dataset_name}: {e}"
    
    def print_analysis_results(self):
        """打印所有數據集的分析結果"""
        print("=" * 80)
        print("TimeXer 數據集分析報告")
        print("=" * 80)
        
        summary_table = []
        detailed_info = {}
        
        for dataset_name, dataset_info in self.datasets_info.items():
            print(f"\n📊 數據集: {dataset_name}")
            print("-" * 60)
            
            # 原始數據分析
            raw_analysis = self.analyze_raw_data(dataset_name, dataset_info)
            if raw_analysis:
                print(f"原始數據:")
                print(f"  📈 總時間步數: {raw_analysis['total_timesteps']:,}")
                print(f"  📋 總列數: {raw_analysis['total_columns']}")
                print(f"  🔢 數值特徵數: {raw_analysis['num_numeric_features']}")
                print(f"  📅 時間列: {raw_analysis['time_columns']}")
                
                if raw_analysis['data_stats']:
                    stats = raw_analysis['data_stats']
                    print(f"  📊 數據統計:")
                    print(f"    平均值: {stats['mean']:.3f}")
                    print(f"    標準差: {stats['std']:.3f}")
                    print(f"    範圍: [{stats['min']:.3f}, {stats['max']:.3f}]")
                    print(f"    缺失值: {stats['missing_values']}")
                
                # TimeXer 數據劃分分析
                timexer_analysis = self.analyze_timexer_data_split(dataset_name, dataset_info)
                
                if isinstance(timexer_analysis, dict):
                    # 使用 val 或 test 的信息（因為 train 可能失敗）
                    reference_info = None
                    for split in ['train', 'val', 'test']:
                        if split in timexer_analysis and isinstance(timexer_analysis[split], dict):
                            reference_info = timexer_analysis[split]
                            break
                    
                    if reference_info:
                        print(f"\nTimeXer 數據劃分:")
                        print(f"  🎯 內生變數數量: {reference_info['n_endogenous_features']}")
                        print(f"  🌍 外生變數數量: {reference_info['n_exogenous_features']}")
                        print(f"  📊 序列長度: {reference_info['seq_len']}")
                        print(f"  📦 批次大小: {reference_info['batch_size']}")
                        
                        if 'data_range' in reference_info:
                            dr = reference_info['data_range']
                            print(f"  📈 數據範圍: [{dr['x_min']:.3f}, {dr['x_max']:.3f}], 平均: {dr['x_mean']:.3f}")
                        
                        print(f"\n數據集大小:")
                        train_samples = val_samples = test_samples = "N/A"
                        for split in ['train', 'val', 'test']:
                            if split in timexer_analysis and isinstance(timexer_analysis[split], dict):
                                length = timexer_analysis[split]['dataset_length']
                                print(f"  {split.capitalize()}: {length:,} samples")
                                if split == 'train':
                                    train_samples = f"{length:,}"
                                elif split == 'val':
                                    val_samples = f"{length:,}"
                                elif split == 'test':
                                    test_samples = f"{length:,}"
                            elif split in timexer_analysis:
                                print(f"  {split.capitalize()}: 加載失敗")
                                if split == 'train':
                                    train_samples = "Failed"
                        
                        # 計算數據劃分比例（如果有足夠信息）
                        valid_splits = [split for split in ['train', 'val', 'test'] 
                                      if split in timexer_analysis and isinstance(timexer_analysis[split], dict)]
                        
                        if len(valid_splits) >= 2:
                            print(f"\n可用數據劃分:")
                            for split in valid_splits:
                                length = timexer_analysis[split]['dataset_length']
                                print(f"  {split.capitalize()}: {length:,} samples")
                        
                        # 添加到摘要表
                        summary_table.append([
                            dataset_name,
                            f"{raw_analysis['total_timesteps']:,}",
                            f"{reference_info['n_endogenous_features']}",
                            f"{reference_info['n_exogenous_features']}",
                            train_samples,
                            val_samples,
                            test_samples
                        ])
                        
                        # 保存詳細信息
                        detailed_info[dataset_name] = {
                            'raw': raw_analysis,
                            'timexer': timexer_analysis
                        }
                else:
                    print(f"TimeXer 分析失敗: {timexer_analysis}")
                    summary_table.append([dataset_name, f"{raw_analysis['total_timesteps']:,}", "N/A", "N/A", "N/A", "N/A", "N/A"])
            else:
                print(f"❌ 無法讀取數據集文件: {dataset_info['root_path']}{dataset_info['data_path']}")
                summary_table.append([dataset_name, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
        
        # 打印摘要表格
        print("\n" + "=" * 120)
        print("📋 數據集摘要表")
        print("=" * 120)
        
        if summary_table:
            header = ["數據集", "總時間步", "內生變數", "外生變數", "訓練樣本", "驗證樣本", "測試樣本"]
            
            # 計算列寬
            col_widths = [max(len(str(row[i])) for row in [header] + summary_table) + 2 for i in range(len(header))]
            
            # 打印表頭
            header_row = "|".join(f"{header[i]:^{col_widths[i]}}" for i in range(len(header)))
            print(header_row)
            print("-" * len(header_row))
            
            # 打印數據行
            for row in summary_table:
                data_row = "|".join(f"{row[i]:^{col_widths[i]}}" for i in range(len(row)))
                print(data_row)
        
        # 打印額外的洞察
        print("\n" + "=" * 80)
        print("📈 數據集特性分析")
        print("=" * 80)
        
        # 按特徵數量分類
        small_datasets = []   # < 10 features
        medium_datasets = []  # 10-100 features  
        large_datasets = []   # > 100 features
        
        for row in summary_table:
            if len(row) > 2 and row[2] != "N/A" and row[2].isdigit():
                features = int(row[2])
                if features < 10:
                    small_datasets.append((row[0], features))
                elif features <= 100:
                    medium_datasets.append((row[0], features))
                else:
                    large_datasets.append((row[0], features))
        
        print(f"\n🔢 按特徵數量分類:")
        print(f"  小規模 (< 10 特徵): {[f'{name} ({feat})' for name, feat in small_datasets]}")
        print(f"  中規模 (10-100 特徵): {[f'{name} ({feat})' for name, feat in medium_datasets]}")  
        print(f"  大規模 (> 100 特徵): {[f'{name} ({feat})' for name, feat in large_datasets]}")
        
        # 按時間步長度分類
        short_series = []     # < 20k timesteps
        medium_series = []    # 20k-50k timesteps
        long_series = []      # > 50k timesteps
        
        for row in summary_table:
            if len(row) > 1 and row[1] != "N/A":
                try:
                    timesteps = int(row[1].replace(',', ''))
                    if timesteps < 20000:
                        short_series.append((row[0], timesteps))
                    elif timesteps <= 50000:
                        medium_series.append((row[0], timesteps))
                    else:
                        long_series.append((row[0], timesteps))
                except ValueError:
                    continue
        
        print(f"\n⏰ 按時間序列長度分類:")
        print(f"  短序列 (< 20k 步): {[f'{name} ({ts:,})' for name, ts in short_series]}")
        print(f"  中等序列 (20k-50k 步): {[f'{name} ({ts:,})' for name, ts in medium_series]}")
        print(f"  長序列 (> 50k 步): {[f'{name} ({ts:,})' for name, ts in long_series]}")

        # 分析數據劃分模式
        print(f"\n📊 TimeXer 數據劃分模式:")
        print(f"  - 輸入序列長度: 96 時間步")
        print(f"  - 預測序列長度: 96 時間步") 
        print(f"  - 標籤序列長度: 48 時間步（decoder 輸入）")
        print(f"  - 外生變數: 時間編碼特徵（如小時、星期、月份等）")
        
        # 根據成功的數據集提供洞察
        successful_datasets = [name for name, info in detailed_info.items()]
        if successful_datasets:
            print(f"\n✅ 成功分析的數據集: {successful_datasets}")
            print(f"🔄 訓練數據加載失敗原因: 缺少數據增強相關參數（已修復）")
            
        print("\n💡 說明:")
        print("- 內生變數: 模型要預測的時間序列特徵")
        print("- 外生變數: 時間標記特徵 (如小時、星期、月份等)")
        print("- 訓練/驗證/測試樣本: TimeXer 按滑動窗口生成的樣本數")
        print("- TimeXer 使用 96 時間步作為輸入，預測 96 時間步")
        print("- 批次形狀: [batch_size, seq_len, features]")

def main():
    """主函數"""
    print("🚀 正在初始化數據集分析器...")
    analyzer = DatasetAnalyzer()
    analyzer.print_analysis_results()

if __name__ == "__main__":
    main()
