"""
RMSE推移の平均化用
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import sem, t
import os
import japanize_matplotlib
# === 設定 ===
input_dir = "/Users/hideyasu/Documents/interpolation_project/pred_Scipy/tempera_prediction/bc_thresh_avg_eval/RESULTS/avg_non-col"  # CSVファイルの格納先
target_thresholds = [5, 4, 3, 2, 1]

rmse_data_per_threshold = {th: [] for th in target_thresholds}

for filename in os.listdir(input_dir):
    if filename.endswith(".csv") and filename.startswith("rmse_seed"):
        df = pd.read_csv(os.path.join(input_dir, filename))
        print(f"読み込み成功: {filename} | 列: {df.columns.tolist()}")
        
        for th in target_thresholds:
            col_name = f"RMSE_thresh{th}"
            if col_name in df.columns:
                rmse_data_per_threshold[th].append(df[col_name])
            else:
                print(f"警告: {col_name} が {filename} に存在しません")

plt.figure(figsize=(8,6))
colors = plt.colormaps.get_cmap('tab10')

for idx, th in enumerate(target_thresholds):
    rmse_trials = rmse_data_per_threshold[th]
    
    if len(rmse_trials) == 0:
        print(f"データなし: 蓄積数={th}")
        continue

    max_len = max(len(r) for r in rmse_trials)
    padded_rmse = np.full((len(rmse_trials), max_len), np.nan)
    
    for i, series in enumerate(rmse_trials):
        padded_rmse[i, :len(series)] = series.values

    mean_rmse = np.nanmean(padded_rmse, axis=0)
    stderr = sem(padded_rmse, axis=0, nan_policy='omit')
    ci95 = t.ppf(0.975, len(rmse_trials) - 1) * stderr

    x = np.arange(len(mean_rmse))
    color = colors(idx % 10)

    plt.plot(x, mean_rmse, label=f"蓄積数={th}", color=color)
    plt.fill_between(x, mean_rmse - ci95, mean_rmse + ci95, color=color, alpha=0.3)

plt.xlabel("累積通知回数")
plt.ylabel("RMSE[℃]")
plt.title("通知数基準ブロードキャスト 平均RMSEと95%信頼区間")
plt.grid()
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
colors = plt.colormaps.get_cmap('tab10')

for idx, th in enumerate(target_thresholds):
    rmse_trials = rmse_data_per_threshold[th]
    
    if len(rmse_trials) == 0:
        print(f"データなし: 蓄積数={th}")
        continue

    max_len = max(len(r) for r in rmse_trials)
    padded_rmse = np.full((len(rmse_trials), max_len), np.nan)
    
    for i, series in enumerate(rmse_trials):
        padded_rmse[i, :len(series)] = series.values

    mean_rmse = np.nanmean(padded_rmse, axis=0)
    stderr = sem(padded_rmse, axis=0, nan_policy='omit')
    ci95 = t.ppf(0.975, len(rmse_trials) - 1) * stderr

    x = np.arange(len(mean_rmse))
    color = colors(idx % 10)

    plt.plot(x, mean_rmse, label=f"蓄積数={th}", color=color)

plt.xlabel("累積通知回数")
plt.ylabel("RMSE[℃]")
plt.title("通知数基準ブロードキャスト 平均RMSEと95%信頼区間")
plt.grid()
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
plt.legend()
plt.show()