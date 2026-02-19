import numpy as np
from scipy.interpolate import SmoothBivariateSpline

class AccessPoint:
    def __init__(self, mesh_size, sensor_network):
        self.mesh_size = mesh_size
        self.sensor_network = sensor_network
        self.estimated_matrix = np.full((mesh_size, mesh_size), np.nan)
        self.notifications = {}

    def receive_notification(self, sensor):
        self.notifications[(sensor.row, sensor.col)] = sensor.true_value

    def perform_interpolation(self):
        sensors = self.sensor_network.sensors

        known_sensors = [
            s for s in sensors 
            if isinstance(s.estimated_value, (int, float, np.floating)) and not np.isnan(s.estimated_value)
        ]

        x_known = np.array([s.row for s in known_sensors])
        y_known = np.array([s.col for s in known_sensors])
        z_known = np.array([s.estimated_value for s in known_sensors])

        if len(z_known) < 16:
            return

        spline = SmoothBivariateSpline(x_known, y_known, z_known, kx=3, ky=3)

        print('$$$$$ 補間直後 各未通知センサの誤差 $$$$$')
        for sensor in sensors:
            if sensor.is_missing():
                interpolated = float(spline(sensor.row, sensor.col))
                sensor.set_estimated_value(interpolated)

        # 補間後、未通知センサの誤差を再計算
        for sensor in sensors:
            if sensor.is_missing():
                sensor.update_error()

        # 再計算後に出力
        for sensor in sensors:
            if sensor.is_missing():
                print(f"センサ ({sensor.row},{sensor.col}) | 誤差: {sensor.error:.2f}")

        # ★ 誤差更新直後にRMSEを算出して返す
        current_rmse = self.compute_rmse()
        print(f"[補間直後RMSE] {current_rmse:.2f}")
        return current_rmse

    def get_estimation_matrix(self):
        return self.estimated_matrix

    def compute_rmse(self):
        errors = []
        for sensor in self.sensor_network.sensors:
            if sensor.is_missing():
                error = sensor.calculate_error()
                if not np.isnan(error):
                    errors.append(error)

        if not errors:
            return 0.0
        return np.sqrt(np.mean(np.square(errors)))
