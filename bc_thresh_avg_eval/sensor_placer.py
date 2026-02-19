from sensor import Sensor, RandomSensor

class SensorPlacer:
    @staticmethod
    def grid_placement(mesh_size, cw_table, deny_list=None, sensor_type="proposal"):
        """
        各マスに1台ずつセンサを配置する。

        Parameters:
        - mesh_size: フィールドの縦横サイズ
        - cw_table: 各センサに持たせるCW対応表
        - deny_list: [(row, col), ...] 通知不可にするセンサリスト（オプション）
        - sensor_type: "proposal" or "random" に応じてセンサ種別を切替

        Returns:
        - センサオブジェクトのリスト
        """
        sensors = []
        for row in range(mesh_size):
            for col in range(mesh_size):
                notify = True
                if deny_list and (row, col) in deny_list:
                    notify = False
                if sensor_type == "proposal":
                    sensors.append(Sensor(row, col, cw_table, notify_enabled=notify))
                elif sensor_type == "random":
                    sensors.append(RandomSensor(row, col, cw_table, notify_enabled=notify))
        return sensors