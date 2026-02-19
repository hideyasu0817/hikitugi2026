import os
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import math
import numpy as np

# === 設定 ===
input_dir = "/Users/hideyasu/Documents/interpolation_project/pred_Scipy/tempera_prediction/bc_thresh_avg_eval/cw_range_log"  # 誤差範囲のcsvファイルが格納されているディレクトリ
target_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

# === ファイルごとに処理 ===
for file in target_files:
    file_path = os.path.join(input_dir, file)
    df = pd.read_csv(file_path)

    # ヘッダー取得
    columns = df.columns.tolist()
    iteration_col = columns[0]
    range_cols = columns[1:]  # 誤差範囲列

    num_iterations = len(df)

    # 最終イテレーションはセンサ数0のため除外
    total_counts = df[range_cols].sum(axis=1)
    valid_indices = total_counts[:-1].index if total_counts.iloc[-1] == 0 else total_counts.index
    num_valid = len(valid_indices)

    if num_valid == 0:
        print(f"{file} は有効なイテレーションがありません。スキップします。")
        continue

    # y軸の最大値取得（自然数の範囲に丸める）
    ymax = int(np.nanmax(df.iloc[valid_indices][range_cols].values)) + 1

    # サブプロットレイアウト
    cols = 4
    rows = math.ceil(num_valid / cols)

    plt.figure(figsize=(4 * cols, 3 * rows))
    plt.suptitle(f"{file} の誤差範囲ごとのセンサ数（各イテレーション）", fontsize=16)

    for plot_idx, row_idx in enumerate(valid_indices):
        counts = df.iloc[row_idx][range_cols].values
        iteration_num = df.iloc[row_idx][iteration_col]

        plt.subplot(rows, cols, plot_idx + 1)
        plt.bar(range_cols, counts)
        plt.title(f"Iter {int(iteration_num)}")
        plt.ylim(0, ymax)
        plt.yticks(np.arange(0, ymax + 1, 1))
        plt.xticks(rotation=45)
        plt.grid(axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
