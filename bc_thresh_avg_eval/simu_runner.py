# simu_runner.py
import numpy as np
import copy
import matplotlib.pyplot as plt

# class SimulationRunner:
#     def __init__(self, heatmap_model, sensor_network, aggregator, error_threshold=1.5, max_iter=50):
#         self.heatmap_model = heatmap_model
#         self.sensor_network = sensor_network
#         self.aggregator = aggregator
#         self.error_threshold = error_threshold
#         self.max_iter = max_iter
#         self.rmse_records = []
#         self.rmse_records_nonrep = []
#         self.max_error_records = []
#         self.initial_missing_positions = [(s.row, s.col) for s in sensor_network.get_missing_sensors()]
#         self.stored_array = None
#         self.notification_trials = []
#         self.notification_log = {}
#         self.collision_count = 0
#         self.broadcast_rmse_records = []
#         self.broadcast_max_error_records = []

#     # def run(self):
#     # #　比較用：①
#     # # ランダム通知/誤差に基づく優先制御通知を比較するときは"比較用：①"を有効化。”比較用：②”のdef run(self)はコメントアウト
#     #     temperature_map = self.heatmap_model.get_current_map()
#     #     self.sensor_network.update_true_values(temperature_map)

#     #     for sensor in self.sensor_network.sensors:
#     #         if sensor.notify_enabled:
#     #             sensor.notify()

#     #     self.aggregator.perform_interpolation(self.sensor_network)
#     #     self.stored_array = copy.deepcopy(self.get_estimated_values_matrix())
#     #     self.update_rmse_records()

#     #     rmse = self.aggregator.compute_rmse(self.sensor_network)
#     #     max_error = self.calculate_max_error_unnotified()
#     #     self.broadcast_rmse_records.append(rmse)
#     #     self.broadcast_max_error_records.append(max_error)
#     #     iteration = 0

#     #     while iteration < self.max_iter:
#     #         iteration += 1
#     #         print(f"\n=== Iteration {iteration} ===")

#     #         self.aggregator.perform_interpolation(self.sensor_network)

#     #         self.aggregator.perform_interpolation(self.sensor_network)
#     #         rmse = self.aggregator.compute_rmse(self.sensor_network)
#     #         max_error = self.calculate_max_error_unnotified()
#     #         self.broadcast_rmse_records.append(rmse)
#     #         self.broadcast_max_error_records.append(max_error)

#     #         print("各未通知センサの予測値/真値/誤差：")
#     #         for sensor in self.sensor_network.get_missing_sensors():
#     #             est = sensor.estimated_value
#     #             true = sensor.true_value
#     #             if not np.isnan(est) and true is not None:
#     #                 error = abs(est - true)
#     #                 print(f" センサ {sensor.row}-{sensor.col} | 推定: {est:.2f}℃ | 真値: {true:.2f}℃ | 誤差: {error:.2f}℃")

#     #         backoff_times = {}
#     #         for sensor in self.sensor_network.get_missing_sensors():
#     #             time = sensor.determine_backoff_time(error_threshold=self.error_threshold)
#     #             if time < np.inf:
#     #                 backoff_times[sensor] = time

#     #         if not backoff_times:
#     #             print("通知対象センサなし。終了。")
#     #             break

#     #         min_time = min(backoff_times.values())
#     #         candidates = [sensor for sensor, time in backoff_times.items() if time == min_time]
#     #         self.notification_trials.append(iteration)

#     #         if len(candidates) > 1:
#     #             print(f"★コリジョン発生！競合センサ: {[f'{s.row}-{s.col}' for s in candidates]}")
#     #             self.collision_count += 1

#     #         else:
#     #             selected_sensor = candidates[0]
#     #             error = abs(selected_sensor.estimated_value - selected_sensor.true_value)
#     #             if error < self.error_threshold:
#     #                 print(f"端末 {selected_sensor.row}-{selected_sensor.col} の誤差 {error:.2f} はerr_lev未満のため未通知")
#     #             else:
#     #                 selected_sensor.notify()
#     #                 print(f" -> {selected_sensor.row}-{selected_sensor.col} の通知が成功！")
#     #                 self.notification_log[(selected_sensor.row, selected_sensor.col)] = iteration
#     #                 self.stored_array[selected_sensor.row, selected_sensor.col] = selected_sensor.true_value

#     #         self.aggregator.perform_interpolation(self.sensor_network)
#     #         self.update_rmse_records()

#     #         rmse = self.rmse_records[-1]
#     #         rmse_nonrep = self.rmse_records_nonrep[-1]
#     #         max_error = self.max_error_records[-1]
#     #         print(f"再補間ありRMSE: {float(rmse):.2f}, 再補間なしRMSE: {float(rmse_nonrep):.2f}, 最大誤差: {float(max_error):.3f}")


#     #         if max_error < self.error_threshold:
#     #             print("全センサ誤差が許容範囲内。終了。")
#     #             break

#     #     print("シミュレーション終了。")

#     def run(self):
#     # 比較用：②
#         print("=== [都度補間戦略] 通知の都度補間・ブロードキャスト ===")

#         # 初期マップの取得と真値の割り当て
#         temperature_map = self.heatmap_model.get_current_map()
#         self.sensor_network.update_true_values(temperature_map)

#         # 初期通知処理（notify_enabledなセンサのみ）
#         for sensor in self.sensor_network.sensors:
#             if sensor.notify_enabled:
#                 sensor.notify()

#         # 初期補間と記録（Iteration 0 相当）
#         self.aggregator.perform_interpolation(self.sensor_network)
#         self.stored_array = copy.deepcopy(self.get_estimated_values_matrix())
#         self.update_rmse_records()

#         init_rmse = self.aggregator.compute_rmse(self.sensor_network)
#         init_max_error = self.calculate_max_error_unnotified()
#         self.broadcast_rmse_records.append(init_rmse)
#         self.broadcast_max_error_records.append(init_max_error)

#         print(f"[Init] RMSE: {init_rmse:.2f}, Max Error: {init_max_error:.2f}")

#         # イテレーションループ（Iteration 1 以降）
#         for iteration in range(1, self.max_iter + 1):
#             print(f"\n=== Iteration {iteration} ===")

#             # バックオフ時間の決定（誤差に基づく通知優先制御）
#             backoff_times = {}
#             for sensor in self.sensor_network.get_missing_sensors():
#                 time = sensor.determine_backoff_time(error_threshold=self.error_threshold)
#                 if time < np.inf:
#                     backoff_times[sensor] = time

#             if not backoff_times:
#                 print("通知対象センサなし。終了。")
#                 break

#             min_time = min(backoff_times.values())
#             candidates = [sensor for sensor, time in backoff_times.items() if time == min_time]

#             if len(candidates) > 1:
#                 print(f"★コリジョン発生！競合センサ: {[f'{s.row}-{s.col}' for s in candidates]}")
#                 self.collision_count += 1
#                 continue  # 通知失敗時はスキップ

#             # 通知成功処理
#             selected_sensor = candidates[0]
#             selected_sensor.notify()
#             self.notification_log[(selected_sensor.row, selected_sensor.col)] = iteration
#             self.stored_array[selected_sensor.row, selected_sensor.col] = selected_sensor.true_value

#             # 補間・評価・記録（通知成功時のみ）
#             self.aggregator.perform_interpolation(self.sensor_network)
#             rmse = self.aggregator.compute_rmse(self.sensor_network)
#             max_error = self.calculate_max_error_unnotified()
#             self.broadcast_rmse_records.append(rmse)
#             self.broadcast_max_error_records.append(max_error)
#             self.update_rmse_records()

#             print(f"RMSE: {rmse:.2f}, Max Error: {max_error:.2f}")

#             if max_error < self.error_threshold:
#                 print("全センサ誤差が許容範囲内。終了。")
#                 break

#         print("シミュレーション終了。")


#     def resolve_collision(self, backoff_times):
#         """
#         バックオフ時間が最小のセンサを選出し、コリジョンを判定。
#         同じ最小時間のセンサが複数いたらコリジョンとして None を返す。
#         """
#         if not backoff_times:
#             return None

#         min_time = min(backoff_times.values())
#         candidates = [sensor for sensor, time in backoff_times.items() if time == min_time]

#         if len(candidates) == 1:
#             return candidates[0]  # 通知成功
#         else:
#             return None  # コリジョン

#     def update_rmse_records(self):
#         rmse = self.calculate_rmse()
#         rmse_nonrep = self.calculate_rmse_from_array(self.stored_array)
#         max_error = self.calculate_max_error_unnotified()
#         self.rmse_records.append(rmse)
#         self.rmse_records_nonrep.append(rmse_nonrep)
#         self.max_error_records.append(max_error)

#     def get_total_notifications(self):
#         return len(self.notification_log)

#     def get_total_collisions(self):
#         return self.collision_count

#     # def calculate_rmse(self):
#     #     errors = []
#     #     for sensor in self.sensor_network.sensors:
#     #         if not np.isnan(sensor.estimated_value) and sensor.true_value is not None:
#     #             errors.append(sensor.estimated_value - sensor.true_value)
#     #     if not errors:
#     #         return np.nan
#     #     return np.sqrt(np.mean(np.square(errors)))
#     def calculate_rmse(self):
#         errors = []
#         for sensor in self.sensor_network.sensors:
#             if not np.isnan(sensor.estimated_value) and sensor.true_value is not None:
#                 est_val = float(sensor.estimated_value[0]) if isinstance(sensor.estimated_value, np.ndarray) else float(sensor.estimated_value)
#                 true_val = float(sensor.true_value)
#                 errors.append(est_val - true_val)
#         if not errors:
#             return np.nan
#         return np.sqrt(np.mean(np.square(errors)))

#     def calculate_rmse_from_array(self, estimated_array):
#         true_array = self.get_true_values_matrix()
#         errors = []
#         for row, col in self.initial_missing_positions:
#             true_val = true_array[row, col]
#             est_val = estimated_array[row, col]
#             errors.append(abs(true_val - est_val))
#         return np.sqrt(np.mean(np.square(errors)))

#     def calculate_max_error(self):
#         errors = []
#         for sensor in self.sensor_network.sensors:
#             if not np.isnan(sensor.estimated_value) and sensor.true_value is not None:
#                 errors.append(abs(sensor.estimated_value - sensor.true_value))
#         if not errors:
#             return np.nan
#         return max(errors)
    
#     def calculate_max_error_unnotified(self):
#         """
#         未通知センサのみを対象とした最大誤差を返す。
#         """
#         errors = []
#         for sensor in self.sensor_network.get_missing_sensors():
#             if not np.isnan(sensor.estimated_value) and sensor.true_value is not None:
#                 errors.append(abs(sensor.estimated_value - sensor.true_value))
#         return max(errors) if errors else np.nan

#     def get_estimated_values_matrix(self):
#         mesh_size = self.sensor_network.get_mesh_size()
#         est_matrix = np.full((mesh_size, mesh_size), np.nan)
#         for sensor in self.sensor_network.sensors:
#             est_matrix[sensor.row, sensor.col] = sensor.estimated_value
#         return est_matrix

#     def get_true_values_matrix(self):
#         mesh_size = self.sensor_network.get_mesh_size()
#         true_matrix = np.full((mesh_size, mesh_size), np.nan)
#         for sensor in self.sensor_network.sensors:
#             if sensor.true_value is not None:
#                 true_matrix[sensor.row, sensor.col] = sensor.true_value
#         return true_matrix

#     def plot_rmse_comparison(runner_rand, runner):
#         # RMSE記録リスト（0番目は初期補間 → 除外）
#         rmse_list_A_with = runner_rand.rmse_records[1:]       # ランダム通知（再補間あり）
#         rmse_list_A_without = runner_rand.rmse_records_nonrep[1:]  # ランダム通知（再補間なし）

#         rmse_list_B_with = runner.rmse_records[1:]            # 提案法（再補間あり）
#         rmse_list_B_without = runner.rmse_records_nonrep[1:]  # 提案法（再補間なし）

#         # 通知試行回数は1から始める（初期補間は除外）
#         iterations_A = list(range(1, len(rmse_list_A_with) + 1))
#         iterations_B = list(range(1, len(rmse_list_B_with) + 1))

#         plt.figure(figsize=(10, 6))
#         plt.plot(iterations_A, rmse_list_A_with, marker='o', linestyle='-', color='#1f77b4', label="ランダム通知 - 再補間あり")
#         plt.plot(iterations_A, rmse_list_A_without, marker='o', linestyle='--', color='#1f77b4', label="ランダム通知 - 再補間なし")
#         plt.plot(iterations_B, rmse_list_B_with, marker='x', linestyle='-', color='red', label="誤差大を優先通知 - 再補間あり")
#         plt.plot(iterations_B, rmse_list_B_without, marker='x', linestyle='--', color='red', label="誤差大を優先通知 - 再補間なし")

#         plt.xlabel("通知試行回数（初期補間は除外）")
#         plt.ylabel("RMSE [℃]")
#         plt.title("通知戦略によるRMSE推移の比較")
#         plt.grid(True)
#         plt.legend()
#         plt.tight_layout()
#         plt.show()
    


class SimulationRunner:
    def __init__(self, heatmap_model, sensor_network, aggregator, max_iter=100, error_threshold=0.5):
        self.heatmap_model = heatmap_model
        self.sensor_network = sensor_network
        self.aggregator = aggregator
        self.max_iter = max_iter
        self.error_threshold = error_threshold
        self.rmse_records = []
        self.rmse_records_nonrep = []
        self.max_error_records = []
        self.notification_log = {}
        self.collision_count = 0
        self.stored_array = None

    def get_estimated_values_matrix(self):
        size = self.sensor_network.grid_size
        matrix = np.full((size, size), np.nan)
        for sensor in self.sensor_network.sensors:
            matrix[sensor.row, sensor.col] = sensor.estimated_value
        return matrix

    def update_rmse_records(self):
        rmse = self.aggregator.compute_rmse()
        self.rmse_records.append(rmse)

        est_matrix = self.get_estimated_values_matrix()
        true_matrix = self.sensor_network.get_true_values_matrix()

        errors = []
        for sensor in self.sensor_network.sensors:
            if sensor.is_missing():
                est = est_matrix[sensor.row, sensor.col]
                true = true_matrix[sensor.row, sensor.col]
                if not np.isnan(est) and true is not None:
                    errors.append(abs(est - true))

        if not errors:
            self.rmse_records_nonrep.append(0.0)
            self.max_error_records.append(0.0)
        else:
            self.rmse_records_nonrep.append(np.sqrt(np.mean(np.square(errors))))
            self.max_error_records.append(max(errors))

    def run(self):
        print("=== [提案法] 誤差優先通知シミュレーション ===")

        temperature_map = self.heatmap_model.get_current_map()
        self.sensor_network.update_true_values(temperature_map)

        for sensor in self.sensor_network.sensors:
            if sensor.notify_enabled:
                sensor.notify()
                self.aggregator.receive_notification(sensor)

        self.stored_array = copy.deepcopy(self.get_estimated_values_matrix())
        self.update_rmse_records()

        for iteration in range(1, self.max_iter + 1):
            print(f"\n=== Iteration {iteration} ===")

            backoff_times = {}
            for sensor in self.sensor_network.get_missing_sensors():
                time = sensor.determine_backoff_time(error_threshold=self.error_threshold)
                if time < np.inf:
                    backoff_times[sensor] = time

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
                print(f" -> {selected_sensor.row}-{selected_sensor.col} 通知成功")
                self.aggregator.receive_notification(selected_sensor)

            self.update_rmse_records()

            rmse = self.rmse_records[-1]
            rmse_nonrep = self.rmse_records_nonrep[-1]
            max_error = self.max_error_records[-1]
            print(f"再補間ありRMSE: {float(rmse):.2f}, 再補間なしRMSE: {float(rmse_nonrep):.2f}, 最大誤差: {float(max_error):.3f}")

            if max_error < self.error_threshold:
                print("全センサ誤差が許容範囲内。終了。")
                break

        print("シミュレーション終了。")