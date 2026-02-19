import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

class HeatmapModel:
    def __init__(self, mesh_size=10, sigma=2, temp_min=10, temp_max=28, seed=None):
        self.mesh_size = mesh_size
        self.sigma = sigma
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.seed = seed
        self.current_map = None
        self.generate_new_map()

    def generate_new_map(self):
        rng = np.random.default_rng(self.seed)  # 局所的な乱数生成器（seed固定）

        num_points = self.mesh_size * self.mesh_size
        x = rng.uniform(0, self.mesh_size, num_points)
        y = rng.uniform(0, self.mesh_size, num_points)
        temperature = rng.uniform(self.temp_min, self.temp_max, num_points)

        grid_temperature = gaussian_filter(
            np.histogram2d(x, y, bins=self.mesh_size, weights=temperature, density=False)[0],
            sigma=self.sigma
        )
        grid_temperature = np.clip(grid_temperature, self.temp_min, self.temp_max)
        self.current_map = np.round(grid_temperature, 3)

    def get_current_map(self):
        return self.current_map


## ヒートマップ確認用
if __name__ == "__main__":
    # インスタンス生成
    heatmap = HeatmapModel(mesh_size=10, sigma=2, temp_min=8, temp_max=25, seed=99)

    # 温度マップ取得
    temperature_map = heatmap.get_current_map()

    print("Temperature Map（℃）:")
    print(temperature_map)

    # ヒートマップ表示
    plt.figure(figsize=(6, 5))
    plt.imshow(
        temperature_map,
        cmap='coolwarm',
        origin='upper',
        interpolation='none'
    )
    plt.colorbar(label="Temperature (°C)")
    plt.title("Generated Temperature Heatmap")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(False)
    plt.tight_layout()
    plt.show()