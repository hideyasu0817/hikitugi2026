import numpy as np
class SensorNetwork:
    def __init__(self, sensors, mesh_size):
        self.sensors = sensors
        self.grid_size = mesh_size

    def get_mesh_size(self):
        """
        センサが配置されているメッシュのサイズ（正方格子前提）を返す。
        """
        return max(max(sensor.row for sensor in self.sensors),
                   max(sensor.col for sensor in self.sensors)) + 1

    def update_true_values(self, temperature_map):
        """
        各センサに温度マップから真値を与える。
        """
        for sensor in self.sensors:
            sensor.sense(temperature_map)

    def get_missing_sensors(self):
        """
        未通知状態のセンサリストを返す。
        """
        return [s for s in self.sensors if s.is_missing()]

    def get_notified_sensors(self):
        """
        通知済みセンサのリストを返す。
        """
        return [s for s in self.sensors if not s.is_missing()]
    
    def get_true_values_matrix(self):
        """
        各センサの真値（実測値）を2次元配列として返す。
        """
        size = self.grid_size
        matrix = np.full((size, size), np.nan)
        for sensor in self.sensors:
            matrix[sensor.row, sensor.col] = sensor.true_value
        return matrix
