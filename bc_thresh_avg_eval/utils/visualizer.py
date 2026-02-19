import matplotlib.pyplot as plt

class Visualizer:
    @staticmethod
    def plot_heatmap(temperature_map, title="Temperature Heatmap"):
        plt.figure(figsize=(6, 5))
        plt.imshow(
            temperature_map,
            cmap='coolwarm',
            origin='upper',
            interpolation='none'
        )
        plt.colorbar(label="Temperature (°C)")
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(False)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_heatmap_with_sensors(temperature_map, sensor_positions, title="Temperature Map with Sensors"):
        """
        temperature_map: 2D array (温度マップ)
        sensor_positions: [(row, col), ...] のリスト
        """
        plt.figure(figsize=(6, 5))
        plt.imshow(
            temperature_map,
            cmap='coolwarm',
            origin='upper',
            interpolation='none'
        )
        plt.colorbar(label="Temperature (°C)")

        for row, col in sensor_positions:
            plt.plot(col, row, 'ko', markersize=5)  # センサ位置に黒丸プロット

        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(False)
        plt.tight_layout()
        plt.show()

def plot_heatmap_with_sensors(temperature_map, sensors, deny_list=None, title="センサ配置付きヒートマップ"):
    plt.figure(figsize=(6, 5))
    plt.imshow(temperature_map, cmap='coolwarm', origin='upper')
    plt.colorbar(label="Temperature (°C)")

    # deny_listをセット型に変換（高速判定用）
    deny_set = set(deny_list) if deny_list else set()

    # センサを2種類に分類して座標取得
    normal_coords = [(s.col, s.row) for s in sensors if (s.row, s.col) not in deny_set]
    deny_coords = [(s.col, s.row) for s in sensors if (s.row, s.col) in deny_set]

    # 通常センサ：黒丸
    if normal_coords:
        xs, ys = zip(*normal_coords)
        plt.scatter(xs, ys, color='black', marker='s', label='センサ')

    # denyセンサ：赤いバツ
    if deny_coords:
        xs_deny, ys_deny = zip(*deny_coords)
        plt.scatter(xs_deny, ys_deny, color='red', marker='x', s=80, label='未通知状態センサ')

    plt.title(title)
    plt.xlabel("X (col)")
    plt.ylabel("Y (row)")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()

