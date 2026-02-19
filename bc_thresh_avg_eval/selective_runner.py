from simu_runner import SimulationRunner
import numpy as np
import cw_table
import copy

class ThresholdBroadcastRunner(SimulationRunner):
    def __init__(self, *args, broadcast_threshold=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.broadcast_threshold = broadcast_threshold
        self.notification_count = 0
        self.broadcast_rmse_records = []
        self.success_iteration_records = []
        self.error_distribution_records = []

    def run(self):
        print("=== [通知数基準] ブロードキャストシミュレーション ===")

        temperature_map = self.heatmap_model.get_current_map()
        self.sensor_network.update_true_values(temperature_map)

        for sensor in self.sensor_network.sensors:
            if sensor.notify_enabled:
                sensor.notify()
                self.aggregator.receive_notification(sensor)
        self.aggregator.perform_interpolation()

        self.stored_array = copy.deepcopy(self.get_estimated_values_matrix())
        self.update_rmse_records()

    
        for iteration in range(1, self.max_iter + 1):
            print(f"\n=== Iteration {iteration} ===")
            backoff_times = {}
            for sensor in self.sensor_network.get_missing_sensors():
                time = sensor.determine_backoff_time(error_threshold=self.error_threshold)
                if time < np.inf:
                    backoff_times[sensor] = time

            print("****** 未通知状態センサ *****")
            for sensor in self.sensor_network.get_missing_sensors():
                est = sensor.estimated_value
                true = sensor.true_value
                err = sensor.calculate_error()
                est_disp = f"{est:.2f}" if est is not None and not np.isnan(est) else "nan"
                true_disp = f"{true:.2f}" if true is not None else "nan"
                err_disp = f"{err:.2f}" if err is not None and not np.isnan(err) else "nan"
                backoff_disp = f"{backoff_times[sensor]}" if sensor in backoff_times else "N/A"
                print(f"センサ ({sensor.row},{sensor.col}) | 真値: {true_disp} | 予測値: {est_disp} | 誤差: {err_disp} | Backoff: {backoff_disp}")

            if not backoff_times:
                print("通知対象センサなし。終了。")
                break

            min_time = min(backoff_times.values())
            candidates = [sensor for sensor, time in backoff_times.items() if time == min_time]

            if len(candidates) > 1:
                print(f"★コリジョン発生！競合センサ: {[f'{s.row}-{s.col}' for s in candidates]}")
                self.collision_count += 1
            else:
                selected_sensor = candidates[0]
                selected_sensor.notify()
                self.notification_log[(selected_sensor.row, selected_sensor.col)] = iteration
                self.stored_array[selected_sensor.row, selected_sensor.col] = selected_sensor.true_value
                print(f" -> 通知成功：センサ ({selected_sensor.row},{selected_sensor.col})")
                self.success_iteration_records.append(iteration) # add
                self.notification_count += 1
                print(f"   現在の通知蓄積数: {self.notification_count} / 閾値: {self.broadcast_threshold}")
                self.aggregator.receive_notification(selected_sensor)
                
                if self.notification_count >= self.broadcast_threshold:
                    print(f"*** 通知が{self.broadcast_threshold}回蓄積、ブロードキャストを実施 ***")
                    rmse_after_broadcast = self.aggregator.perform_interpolation()
                    print(f"[ブロードキャスト直後RMSE]: {rmse_after_broadcast:.2f}")
                    self.broadcast_rmse_records.append(rmse_after_broadcast)
                    self.notification_count = 0
            self.update_rmse_records()

            rmse = self.rmse_records[-1]
            rmse_nonrep = self.rmse_records_nonrep[-1]
            max_error = self.max_error_records[-1]
            print(f"再補間ありRMSE: {float(rmse):.2f}, 再補間なしRMSE: {float(rmse_nonrep):.2f}, 最大誤差: {float(max_error):.2f}")

            # 誤差範囲ごとのセンサ数カウント
            error_counts = [0] * len(cw_table.get_cw_table())
            for sensor in self.sensor_network.get_missing_sensors():
                error = sensor.calculate_error()
                for idx, (lower, upper, _) in enumerate(cw_table.get_cw_table()):
                    if lower <= error < upper:
                        error_counts[idx] += 1
                        break
            self.error_distribution_records.append([iteration] + error_counts)

            if max_error < self.error_threshold:
                print("全センサ誤差が許容範囲内。終了。")
                break

        print("シミュレーション終了。")

