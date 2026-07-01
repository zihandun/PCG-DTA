import sys
import os
import time
import torch
import torch.nn as nn
import numpy as np
from lifelines.utils import concordance_index

from utils import *
from models.PCGDTA import PCGDTA


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))

def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    return 1 - (upp / float(down))

def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)

def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)
    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))

DATASETS = ['davis','kiba']
MODEL_CLASS = PCGDTA
SETTINGS = ['cold_drug', 'cold_traget', 'cold_target_drug']

# Hyperparameters
TRAIN_BATCH_SIZE = 256 # Davis: 256 KIBA: 256
TEST_BATCH_SIZE = 256
LR = 0.0005  # 0.0005
NUM_EPOCHS = 2000
PATIENCE = 100
SEED = 42

THRESHOLDS = {'davis': 7.0, 'kiba': 12.1}

cuda_name = "cuda:0"
if len(sys.argv) > 1:
    cuda_name = "cuda:" + str(int(sys.argv[1]))
device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

if not os.path.exists('saved_models'): os.makedirs('saved_models')
if not os.path.exists('results'): os.makedirs('results')
if not os.path.exists('data'): os.makedirs('data')

for dataset in DATASETS:
    current_threshold = THRESHOLDS.get(dataset, 7.0)

    for setting in SETTINGS:
        print('\n' + '='*60)
        print(f'Task: {dataset} | Setting: {setting} | Seed: {SEED}')
        print('='*60)

        prefix = f"data/{dataset}_{setting}_{SEED}"
        f_train = f"{prefix}_{prefix}_{setting}_train_{SEED}.data"
        f_val   = f"{prefix}_{prefix}_{setting}_val_{SEED}.data"
        f_test  = f"{prefix}_{prefix}_{setting}_test_{SEED}.data"

        if os.path.isfile(f_train) and os.path.isfile(f_val) and os.path.isfile(f_test):
            print(f'Loading existing data files from {prefix}...')
            train_data = torch.load(f_train)
            val_data   = torch.load(f_val)
            test_data  = torch.load(f_test)
        else:
            print(f'Files not found. Please run create_data_csv.py firstly! ')
            assert False

        print(f'Data sizes -> Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}')

        # DataLoader
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate, num_workers=0)
        val_loader   = torch.utils.data.DataLoader(val_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=0)
        test_loader  = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=0)

        # Model
        model = MODEL_CLASS().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        
        model_path = f'saved_models/model_{MODEL_CLASS.__name__}_{dataset}_{setting}_{SEED}.model'
        res_path   = f'results/res_{MODEL_CLASS.__name__}_{dataset}_{setting}_{SEED}.csv'
        print(model_path)

        # Loop
        best_mse = float('inf')
        best_ci = 0
        best_epoch = -1
        patience_cnt = 0
        
        test_mse_best = 0
        test_ci_best = 0
        test_rm2_best = 0

        print(f"Start Training... Total Epochs: {NUM_EPOCHS}")

        for epoch in range(NUM_EPOCHS):
            t_start = time.time()
            
            train_loss, reg_loss = train(model, device, train_loader, optimizer, epoch+1, dataset=train_data, threshold=current_threshold)
            
            G_val, P_val = predicting(model, device, val_loader)
            val_mse = mse(G_val, P_val)
            val_ci = concordance_index(G_val, P_val)

            t_end = time.time()
            duration = t_end - t_start

            if val_mse < best_mse:
                best_mse = val_mse
                best_ci = val_ci
                best_epoch = epoch + 1
                patience_cnt = 0
                
                torch.save(model.state_dict(), model_path)
                
                G_test, P_test = predicting(model, device, test_loader)
                test_mse_best = mse(G_test, P_test)
                test_ci_best = concordance_index(G_test, P_test)
                test_rm2_best = get_rm2(G_test, P_test)
                
                with open(res_path, 'w') as f:
                    f.write('best_epoch,val_mse,val_ci,test_mse,test_ci,test_rm2\n')
                    f.write(f'{best_epoch},{best_mse:.5f},{best_ci:.5f},{test_mse_best:.5f},{test_ci_best:.5f},{test_rm2_best:.5f}')
                
                print(f'Epoch {epoch+1:04d} | Time: {duration:.1f}s | Loss: {train_loss:.5f} | Val MSE: {val_mse:.5f} | Test MSE: {test_mse_best:.5f} | Test CI: {test_ci_best:.5f} | Test Rm2: {test_rm2_best:.5f} (*)')
            
            else:
                patience_cnt += 1
                print(f'Epoch {epoch+1:04d} | Time: {duration:.1f}s | Loss: {train_loss:.5f} | Val MSE: {val_mse:.5f}')
            
            if patience_cnt >= PATIENCE:
                print(f'\nEarly stopping at epoch {epoch+1}.')
                break
        
        print(f"\n[Result] Setting: {setting}")
        print(f"Best Epoch: {best_epoch}")
        print(f"Val MSE: {best_mse:.4f}")
        print(f"Test MSE: {test_mse_best:.4f}, CI: {test_ci_best:.4f}, Rm2: {test_rm2_best:.4f}")