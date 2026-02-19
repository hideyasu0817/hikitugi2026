"""グラフの出力用 ユーティリティ"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def plot_broadcast_rmse(rmse_lists, max_error_lists, labels):
    # --- グラフ①: RMSE ---
    colors = ['tab:blue', 'tab:red']
    
    plt.figure(figsize=(10, 5))
    for idx, (rmse, label) in enumerate(zip(rmse_lists, labels)):
        x = np.arange(1, len(rmse) + 1)
        plt.plot(x, rmse, marker='o', linestyle='-', label=label, color=colors[idx])

    plt.xlabel("ブロードキャスト回数", fontsize='15')
    plt.ylabel("RMSE [℃]", fontsize='15')
    plt.title("ブロードキャスト単位のRMSE推移", fontsize='15')
    plt.grid(True)
    plt.legend(fontsize='13.5')
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()