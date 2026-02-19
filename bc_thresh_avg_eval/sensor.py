import numpy as np

class Sensor:
    def __init__(self, row, col, cw_table, notify_enabled=True):
        self.row = row
        self.col = col
        self.true_value = None
        self.estimated_value = None
        self.error = np.nan  # 初期状態では誤差は未定義
        self.notified = False
        self.cw_table = cw_table
        self.notify_enabled = notify_enabled

    def sense(self, temperature_map):
        self.true_value = temperature_map[self.row, self.col]

    def set_estimated_value(self, value):
        self.estimated_value = float(value[0]) if isinstance(value, np.ndarray) else float(value)
        self.error = np.nan  # 補間値だけセット、誤差は未定義にリセット

    def update_error(self):
        """
        真値と推定値から誤差を再計算する
        """
        if self.true_value is not None and not np.isnan(self.estimated_value):
            self.error = abs(self.true_value - self.estimated_value)
        else:
            self.error = np.nan

    def notify(self):
        self.estimated_value = self.true_value
        self.error = 0.0  # 通知された場合、誤差はゼロ
        self.notified = True

    def is_missing(self):
        return not self.notified

    def calculate_error(self):
        return self.error

    def determine_backoff_time(self, error_threshold=1.5):
        if not self.is_missing():
            return np.inf

        error = self.error
        if np.isnan(error) or error < error_threshold:
            return np.inf

        cw_range = None
        for low, high, cw_val_range in self.cw_table:
            if low <= error < high:
                cw_range = cw_val_range
                break

        if cw_range is None:
            cw_range = min(
                (cw for _, _, cw in self.cw_table),
                key=lambda r: r[1] - r[0]
            )

        return np.random.randint(cw_range[0], cw_range[1] + 1)



class RandomSensor(Sensor):
    def __init__(self, row, col, cw_table, notify_enabled=True):
        super().__init__(row, col, cw_table, notify_enabled)
        self.fixed_cw = 45  # 固定のコンテンドウィンドウ

    def determine_backoff_time(self, error_threshold=1.5):
        if not self.is_missing():
            return np.inf

        error = self.error
        if np.isnan(error) or error < error_threshold:
            return np.inf

        return np.random.randint(0, self.fixed_cw)
