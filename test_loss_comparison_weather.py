#!/usr/bin/env python3
"""
Loss Function Comparison Experiment on Weather Dataset
比較不同loss函數在Weather dataset上的實際訓練性能

測試的loss函數:
1. MSE (預設)
2. SpikeAwareLoss (基礎spike-aware loss)
3. AdaptiveSpikeAwareLoss (自適應spike-aware loss)
4. HybridLoss (混合loss)
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import time
from datetime import datetime
import json

# Add project root to path
sys.path.append('/home/vm230705/dlp/Project')

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.losses import SpikeAwareLoss, AdaptiveSpikeAwareLoss, HybridLoss
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

class LossComparisonExperiment:
    """Loss函數比較實驗類"""
    
    def __init__(self, base_args):
        self.base_args = base_args
        self.results = {}
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 創建實驗記錄目錄
        self.log_dir = f"/home/vm230705/dlp/Project/loss_comparison_logs_{self.experiment_id}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        print(f"🚀 Loss Comparison Experiment Started: {self.experiment_id}")
        print(f"📁 Logs will be saved to: {self.log_dir}")
    
    def create_loss_functions(self):
        """創建不同的loss函數"""
        loss_functions = {
            # 'MSE': nn.MSELoss(),
            'SpikeAware': SpikeAwareLoss(
                base_loss='mse', 
                spike_weight=2.0, 
                derivative_weight=1.5,
                percentile=85
            ),
            'AdaptiveSpikeAware': AdaptiveSpikeAwareLoss(
                base_loss='mse',
                initial_spike_weight=2.0,
                initial_derivative_weight=1.5
            ),
            'HybridLoss': HybridLoss(
                base_loss='mse',
                alpha=0.7,
                spike_weight=2.0,
                derivative_weight=1.5
            )
        }
        return loss_functions
    
    def run_experiment_with_loss(self, loss_name, loss_function, max_epochs=15):
        """使用指定loss函數運行實驗"""
        print(f"\n🔬 Testing Loss Function: {loss_name}")
        print("=" * 60)
        
        # 複製基礎參數
        args = argparse.Namespace(**vars(self.base_args))
        args.loss = loss_name
        args.train_epochs = max_epochs
        args.des = f'LossComp_{loss_name}_{self.experiment_id}'
        
        # 創建實驗實例
        exp = Exp_Long_Term_Forecast(args)
        
        # 替換loss函數
        exp._select_criterion = lambda: loss_function
        
        # 記錄開始時間
        start_time = time.time()
        
        try:
            print(f"📊 Training with {loss_name} loss...")
            
            # 獲取數據加載器
            train_data, train_loader = exp._get_data(flag='train')
            vali_data, vali_loader = exp._get_data(flag='val')
            test_data, test_loader = exp._get_data(flag='test')
            
            # 訓練過程
            train_metrics = self._custom_train(
                exp, train_loader, vali_loader, test_loader, 
                loss_function, max_epochs, loss_name
            )
            
            # 測試最終性能
            test_metrics = self._evaluate_model(exp, test_loader, loss_function)
            
            training_time = time.time() - start_time
            
            # 記錄結果
            result = {
                'loss_name': loss_name,
                'training_time': training_time,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'final_train_loss': train_metrics['final_train_loss'],
                'final_val_loss': train_metrics['final_val_loss'],
                'epochs_trained': train_metrics['epochs_trained'],
                'early_stopped': train_metrics['early_stopped']
            }
            
            self.results[loss_name] = result
            
            print(f"✅ {loss_name} - Training completed!")
            print(f"   📈 Final Test MSE: {test_metrics['mse']:.6f}")
            print(f"   📉 Final Test MAE: {test_metrics['mae']:.6f}")
            print(f"   ⏱️  Training Time: {training_time:.2f}s")
            print(f"   🔄 Epochs: {train_metrics['epochs_trained']}")
            
            # 保存詳細結果
            self._save_result(loss_name, result)
            
            return result
            
        except Exception as e:
            print(f"❌ Error with {loss_name}: {str(e)}")
            error_result = {
                'loss_name': loss_name,
                'error': str(e),
                'training_time': time.time() - start_time
            }
            self.results[loss_name] = error_result
            return error_result
    
    def _custom_train(self, exp, train_loader, vali_loader, test_loader, criterion, max_epochs, loss_name):
        """自定義訓練過程"""
        model = exp.model
        device = exp.device
        
        # 優化器
        model_optim = exp._select_optimizer()
        
        # Early stopping
        early_stopping = EarlyStopping(patience=3, verbose=True)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(max_epochs):
            model.train()
            epoch_train_loss = []
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                
                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -exp.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :exp.args.label_len, :], dec_inp], dim=1).float().to(device)
                
                # Forward pass
                if exp.args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if exp.args.features == 'MS' else 0
                outputs = outputs[:, -exp.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -exp.args.pred_len:, f_dim:].to(device)
                
                # 計算loss
                if hasattr(criterion, 'forward'):
                    # 對於spike-aware loss，需要傳入預測和真實值
                    loss = criterion(outputs, batch_y)
                else:
                    # 對於標準MSE loss
                    loss = criterion(outputs, batch_y)
                
                epoch_train_loss.append(loss.item())
                
                loss.backward()
                model_optim.step()
            
            # 驗證
            val_loss = self._validate(exp, vali_loader, criterion)
            
            avg_train_loss = np.mean(epoch_train_loss)
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            
            print(f"   Epoch {epoch+1:2d}/{max_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            # Early stopping check
            early_stopping(val_loss, model, f"{self.log_dir}")
            if early_stopping.early_stop:
                print(f"   Early stopping at epoch {epoch+1}")
                break
            
            # 學習率調整
            adjust_learning_rate(model_optim, epoch + 1, exp.args)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'epochs_trained': len(train_losses),
            'early_stopped': early_stopping.early_stop
        }
    
    def _validate(self, exp, vali_loader, criterion):
        """驗證過程"""
        model = exp.model
        device = exp.device
        
        model.eval()
        total_loss = []
        
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                
                dec_inp = torch.zeros_like(batch_y[:, -exp.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :exp.args.label_len, :], dec_inp], dim=1).float().to(device)
                
                if exp.args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if exp.args.features == 'MS' else 0
                outputs = outputs[:, -exp.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -exp.args.pred_len:, f_dim:].to(device)
                
                # 為了公平比較，驗證時都使用MSE
                loss = nn.MSELoss()(outputs, batch_y)
                total_loss.append(loss.item())
        
        return np.mean(total_loss)
    
    def _evaluate_model(self, exp, test_loader, criterion):
        """評估模型最終性能"""
        model = exp.model
        device = exp.device
        
        model.eval()
        preds = []
        trues = []
        
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                
                dec_inp = torch.zeros_like(batch_y[:, -exp.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :exp.args.label_len, :], dec_inp], dim=1).float().to(device)
                
                if exp.args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if exp.args.features == 'MS' else 0
                outputs = outputs[:, -exp.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -exp.args.pred_len:, f_dim:].to(device)
                
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        # 計算標準metrics
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'mspe': mspe
        }
    
    def _save_result(self, loss_name, result):
        """保存實驗結果"""
        result_file = os.path.join(self.log_dir, f"{loss_name}_result.json")
        with open(result_file, 'w') as f:
            # 轉換numpy類型為Python原生類型
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            json.dump(convert_numpy(result), f, indent=2)
    
    def run_all_experiments(self):
        """運行所有loss函數的比較實驗"""
        print(f"🔬 Starting Comprehensive Loss Function Comparison")
        print(f"📊 Dataset: Weather (21 variables, 96-step forecasting)")
        print("=" * 80)
        
        loss_functions = self.create_loss_functions()
        
        for loss_name, loss_func in loss_functions.items():
            try:
                self.run_experiment_with_loss(loss_name, loss_func)
                # 清理GPU緩存
                torch.cuda.empty_cache()
                time.sleep(2)  # 給系統一些休息時間
            except Exception as e:
                print(f"❌ Failed to run experiment with {loss_name}: {str(e)}")
                continue
        
        # 生成比較報告
        self.generate_comparison_report()
    
    def generate_comparison_report(self):
        """生成比較報告"""
        print("\n" + "=" * 80)
        print("📋 LOSS FUNCTION COMPARISON REPORT")
        print("=" * 80)
        
        if not self.results:
            print("❌ No results to compare")
            return
        
        # 創建比較表格
        print(f"{'Loss Function':<20} {'Test MSE':<12} {'Test MAE':<12} {'Time(s)':<10} {'Epochs':<8} {'Early Stop':<12}")
        print("-" * 80)
        
        best_mse = float('inf')
        best_mae = float('inf')
        best_loss_mse = ""
        best_loss_mae = ""
        
        for loss_name, result in self.results.items():
            if 'error' in result:
                print(f"{loss_name:<20} {'ERROR':<12} {'ERROR':<12} {result['training_time']:<10.1f} {'N/A':<8} {'N/A':<12}")
                continue
            
            test_mse = result['test_metrics']['mse']
            test_mae = result['test_metrics']['mae']
            training_time = result['training_time']
            epochs = result['epochs_trained']
            early_stop = 'Yes' if result['early_stopped'] else 'No'
            
            print(f"{loss_name:<20} {test_mse:<12.6f} {test_mae:<12.6f} {training_time:<10.1f} {epochs:<8} {early_stop:<12}")
            
            if test_mse < best_mse:
                best_mse = test_mse
                best_loss_mse = loss_name
            
            if test_mae < best_mae:
                best_mae = test_mae
                best_loss_mae = loss_name
        
        print("\n🏆 WINNERS:")
        print(f"   Best MSE: {best_loss_mse} ({best_mse:.6f})")
        print(f"   Best MAE: {best_loss_mae} ({best_mae:.6f})")
        
        # 保存完整報告
        report_file = os.path.join(self.log_dir, "comparison_report.json")
        with open(report_file, 'w') as f:
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
                
            json.dump({
                'experiment_id': self.experiment_id,
                'best_mse_loss': best_loss_mse,
                'best_mse_value': best_mse,
                'best_mae_loss': best_loss_mae,
                'best_mae_value': best_mae,
                'results': convert_numpy(self.results)
            }, f, indent=2)
        
        print(f"\n📁 Full results saved to: {self.log_dir}")

def create_weather_args():
    """創建Weather dataset的實驗參數"""
    args = argparse.Namespace()
    
    # 基本設置
    args.task_name = 'long_term_forecast'
    args.is_training = 1
    args.model_id = 'weather_96_96_loss_comparison'
    args.model = 'TimeXer'
    args.data = 'custom'
    args.root_path = './dataset/weather/'
    args.data_path = 'weather.csv'
    args.features = 'M'
    
    # 序列參數
    args.seq_len = 96
    args.label_len = 48
    args.pred_len = 96
    
    # 模型參數 (基於Weather dataset的最佳配置)
    args.d_model = 256
    args.d_ff = 512
    args.n_heads = 8
    args.e_layers = 1
    args.d_layers = 1
    args.factor = 3
    args.enc_in = 21
    args.dec_in = 21
    args.c_out = 21
    
    # 訓練參數
    args.batch_size = 4
    args.learning_rate = 0.0001
    args.train_epochs = 10
    args.patience = 3
    args.use_amp = False
    args.lradj = 'type1'
    
    # 其他參數
    args.embed = 'timeF'
    args.freq = 't'
    args.dropout = 0.1
    args.des = 'LossComparison'
    args.use_gpu = True
    args.gpu = 0
    args.use_multi_gpu = False
    args.devices = '0'
    args.checkpoints = './checkpoints/'
    args.output_attention = False
    args.inverse = False
    
    # TimeXer特定參數
    args.patch_len = 16
    args.use_multi_scale = False
    args.use_norm = True  # 添加缺失的參數
    args.activation = 'gelu'  # 添加激活函數參數
    
    # 新增其他必要參數
    args.target = 'OT'
    args.seasonal_patterns = 'Monthly'
    args.num_workers = 0
    args.itr = 1
    args.loss = 'MSE'
    args.expand = 2
    args.d_conv = 4
    args.distil = True
    args.augmentation_ratio = 0  # 添加數據增強參數，設為0表示不使用增強
    
    return args

def main():
    print("🚀 Loss Function Comparison Experiment for Weather Dataset")
    print("📊 Testing: MSE vs SpikeAware vs AdaptiveSpikeAware vs HybridLoss")
    
    # 設置隨機種子
    torch.manual_seed(2021)
    np.random.seed(2021)
    
    # 創建實驗參數
    args = create_weather_args()
    
    # 創建並運行實驗
    experiment = LossComparisonExperiment(args)
    experiment.run_all_experiments()

if __name__ == "__main__":
    main()