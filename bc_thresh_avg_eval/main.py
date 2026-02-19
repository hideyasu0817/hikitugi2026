"""
📁 フォルダ説明：通知数基準のブロキャスシミュレーション。RMSE推移の平均化検証用。

[シミュレータの変更ログ]
2025/05/02：simu_runner.pyはセンサから通知されたら集約局はその都度補間し結果をブロードキャスト
2025/05/14：集約局からブロードキャストする基準を設けた検証用に実装。誤差改善率に基づく通知基準の設定を実装。→ WSNの構成上、集約局が”誤差改善率”を計算するのは無理。（selective_runner.py）
2025/05/23：simu_runner.pyに通知成功時のみ補間・ブロキャス・RMSE算出。元の”whileループ.ver”は、補間と記録が通知に関係なく無条件に毎回実行される。

【問題点】
✅ 最大誤差のカウント・プロットが正しく行えてない。［済］
✅ コリジョン時も通知試行回数としてカウントしたい。［済］
"""

import numpy as np
import matplotlib.pyplot as plt
from heatmap_model import HeatmapModel
from sensor_network import SensorNetwork
from AP import AccessPoint
from cw_table import get_cw_table
from sensor_placer import SensorPlacer
import japanize_matplotlib
import matplotlib.ticker as ticker
import os
import pandas as pd
import numpy as np
import csv
from selective_runner import ThresholdBroadcastRunner
from utils.plot_utils import plot_broadcast_rmse

avg_output_dir = "/Users/hideyasu/Documents/interpolation_project/pred_Scipy/tempera_prediction/bc_thresh_avg_eval/RESULTS/avg_col"
os.makedirs(avg_output_dir, exist_ok=True)

cw_output_dir = "/Users/hideyasu/Documents/interpolation_project/pred_Scipy/tempera_prediction/bc_thresh_avg_eval/RESULTS/cw_range_log"
os.makedirs(cw_output_dir, exist_ok=True)

mesh_size = 10
deny_list = [
    (2,2), (2,3), (2,4),
    (3,2), (3,3), (3,4),
    (4,2), (4,3), (4,4),
    (5,2), (5,3), (5,4),
    (6,6), (6,7), (6,8),
    (8,5),
    (9,5)
]
cw_table = get_cw_table()

# 蓄積数のパターン
threshold_list = [5, 4, 3, 2, 1]
num_trials = 10

# 結果格納用
total_rmse_records = {th: {} for th in threshold_list}  # 通知成功ごとのRMSE推移
total_broadcast_records = {}  # ブロードキャスト時のRMSE推移
total_success_iterations  = {th: {} for th in threshold_list}

# # シミュレーション実行
np.random.seed(123)
# runner_dict = {}

for trial in range(num_trials):
    print(f"\n***** シード {trial} のシミュレーション開始 *****")

    rmse_dict = {f"RMSE_thresh{th}": [] for th in threshold_list}
    max_len = 0

    for threshold in threshold_list:
        print(f"\n=== 蓄積数 {threshold} のシミュレーション開始 ===")
        heatmap = HeatmapModel(mesh_size, sigma=1, temp_min=8, temp_max=25, seed=99)
        sensors = SensorPlacer.grid_placement(mesh_size, cw_table, deny_list=deny_list)
        sensor_network = SensorNetwork(sensors, mesh_size)
        sensor_network.update_true_values(heatmap.get_current_map())
        aggregator = AccessPoint(mesh_size, sensor_network)

        for sensor in sensor_network.sensors:
            if sensor.notify_enabled:
                sensor.notify()
                aggregator.receive_notification(sensor)

        runner = ThresholdBroadcastRunner(heatmap, sensor_network, aggregator, broadcast_threshold=threshold)
        runner.run()
        
        total_rmse_records[threshold] = runner.rmse_records.copy()
        total_broadcast_records[threshold] = runner.broadcast_rmse_records.copy()
        total_success_iterations[threshold] = runner.success_iteration_records.copy()
        # runner_dict[threshold] = runner
        rmse_list = runner.rmse_records
        rmse_dict[f"RMSE_thresh{threshold}"] = rmse_list
        max_len = max(max_len, len(rmse_list))
        error_dist_records = runner.error_distribution_records  # 誤差範囲ごとのセンサ数記録{runner側で事前に記録されている前提}

    for key in rmse_dict:
        rmse_arr = np.array(rmse_dict[key])
        if len(rmse_arr) < max_len:
            rmse_arr = np.pad(rmse_arr, (0, max_len - len(rmse_arr)), constant_values=np.nan)
        rmse_dict[key] = rmse_arr

    df = pd.DataFrame(rmse_dict)
    df.insert(0, "iteration", range(max_len))

    save_path = os.path.join(avg_output_dir, f"rmse_seed{trial}.csv")
    df.to_csv(save_path, index=False, float_format="%.4f")


# ① 通知成功ごとのRMSE推移をまとめて描画
plt.figure(figsize=(8,6))
for threshold in threshold_list:
    rmse_list = total_rmse_records[threshold]
    x = list(range(0, len(rmse_list)))
    plt.plot(x, rmse_list, label=f"蓄積数={threshold}")

plt.xlabel("累積通知回数")
plt.ylabel("RMSE[℃]")
plt.title("通知数基準ブロードキャスト（BC）のRMSE推移比較")
plt.grid()
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
plt.legend()
plt.show()

# plt.figure(figsize=(8,6))
# colors = plt.cm.get_cmap('tab10')

# コリジョン含む
plt.figure(figsize=(8,6))
colors = plt.cm.get_cmap('tab10')

for idx, threshold in enumerate(threshold_list):
    rmse_list = total_rmse_records[threshold]
    x = list(range(0, len(rmse_list)))

    # カラーマップから色を取得
    color = colors(idx % 10)

    # 通常のRMSE推移
    line, = plt.plot(x, rmse_list, label=f"蓄積数={threshold}", color=color)

    # ブロードキャスト時のRMSEにマーカーをつける
    broadcast_indices = []
    notification_count = 0
    success_iter_set = set(total_success_iterations[threshold])

    for i in range(len(rmse_list)):
        if (i+1) in success_iter_set:   
            notification_count += 1
            if notification_count % threshold == 0:
                broadcast_indices.append(i)

    broadcast_x = [x[i] for i in broadcast_indices]
    broadcast_y = [rmse_list[i] for i in broadcast_indices]

    plt.scatter(broadcast_x, broadcast_y, color=color, marker='o', edgecolor='black', label=f"BC発生")

plt.xlabel("累積通知回数")
plt.ylabel("RMSE[℃]")
plt.title("通知数基準ブロードキャスト（BC）のRMSE推移比較")
plt.grid()
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
plt.legend()
plt.show()

# コリジョンなし
plt.figure(figsize=(8, 6))
colors = plt.cm.get_cmap('tab10')

for idx, threshold in enumerate(threshold_list):
    rmse_list = total_rmse_records[threshold]
    success_iters = total_success_iterations[threshold]

    # RMSEと通知成功回数を紐づけてプロット
    x = list(range(1, len(success_iters) + 1))  # 通知成功回数ベース
    y = [rmse_list[i] for i in success_iters]   # 成功した時点のRMSEのみ抽出

    color = colors(idx % 10)
    plt.plot(x, y, marker='o', label=f"蓄積数={threshold}", color=color)

plt.xlabel("通知成功回数")
plt.ylabel("RMSE[℃]")
plt.title("コリジョン除外後のRMSE推移（通知成功回数ベース）")
plt.grid()
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.legend()
plt.tight_layout()
plt.show()

# region
# plt.figure(figsize=(8,6))
# colors = plt.cm.get_cmap('tab10')

# for idx, threshold in enumerate(threshold_list):
#     runner = runner_dict[threshold]
#     rmse_list = total_rmse_records[threshold]
#     success_iters = runner.success_iteration_records

#     # 通知成功ごとのx軸（通知成功回数）
#     x = list(range(1, len(success_iters) + 1))

#     # 通知成功時のRMSEのみ抽出
#     y = [rmse_list[i-1] for i in success_iters]  # i-1に注意（イテレーションは1始まり）

#     color = colors(idx % 10)

#     plt.plot(x, y, label=f"蓄積数={threshold}", color=color)

#     ### ブロードキャスト時のマーカー ###
#     broadcast_indices = []
#     for i in range(len(x)):
#         if (i+1) % threshold == 0:    
#             broadcast_indices.append(i)

#     broadcast_x = [x[i] for i in broadcast_indices]
#     broadcast_y = [y[i] for i in broadcast_indices]

#     plt.scatter(broadcast_x, broadcast_y, color=color, marker='o', edgecolor='black', label=f"BC発生")

# plt.xlabel("累積通知成功回数")  # コリジョンを含まない
# plt.ylabel("RMSE")
# plt.title("通知成功数ベースのRMSE推移比較")
# plt.grid()
# plt.legend()
# plt.show()
# endregion