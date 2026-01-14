
# pq_r=50_l1.log：12:15分钟
# pq_r=50_l2.log：19:55分钟


import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product
from sklearn.model_selection import train_test_split
import logging
from sklearn.model_selection import KFold
# 训练集和测试集

# Logging configuration
logging.basicConfig(filename='./实验结果/pq_r=50_l1_test.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Constants

LIVE_DATA_PATH = 'D:/研究生/毕业论文/代码/网络构建/直播平台网络模型/jupyter/处理数据/processed_hourly_stats_level_1.csv'
CHANGE_DATA_PATH = 'D:/研究生/毕业论文/代码/网络构建/直播平台网络模型/jupyter/处理数据/change_rate_real_time_user_level_1.csv'

DESIRED_COLUMNS = ['real_time_user', 'leave_user', 'user_count']
DESIRED_COLUMNS1 = [f'{col}_t' for col in DESIRED_COLUMNS]
MAX_TIME_POINT = 35
PARAM_VALUES_P = np.arange(0, 1.1, step=0.1).round(1)
PARAM_VALUES_Q = np.arange(-2, 2.1, step=0.5).round(1)
RANDOM_SEED = 50

# Load data
live_data_distribution = pd.read_csv(LIVE_DATA_PATH, index_col=0)
live_data_distribution = live_data_distribution.astype(int)
Nt_change = pd.read_csv(CHANGE_DATA_PATH, index_col=0)

# Generate time points
time_points = np.arange(0, MAX_TIME_POINT)

# Prepare train and test datasets
train_datasets = []
test_datasets = []

for t in time_points:
    data = live_data_distribution[live_data_distribution.index == t]
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=RANDOM_SEED)
    train_datasets.append(train_data)
    test_datasets.append(test_data)


# Function to simulate audience increment
def simulate_audience_increment(distribution_x, change_data, t, p, q):

    increment_total_user = change_data.iloc[t, 0]#iloc使用整数索引，而不是列名
    # total_change = int(increment_total_user * (1+increment_total_user))
    total_change = int(distribution_x.iloc[0] * (1 + increment_total_user))
    all_audience = int(total_change + distribution_x.iloc[2])

    # 初始化增量分布
    increment_distribution = [0] * len(distribution_x)
    increment_distribution[0] = total_change

    # 随机加入总观众数
    random_join_total = int(all_audience * p)
    # 计算每个社区的加入观众数
    random_join_community1 = random_join_total * 0.5
    random_join_community2 = random_join_total - random_join_community1
    # 更新社区1的观众数
    increment_distribution[1] += random_join_community1
    # 更新社区2的观众数
    increment_distribution[2] += random_join_community2

    probabilities = np.array([distribution_x.iloc[i] ** q for i in range(1, 3)])
    # # 将inf对应的概率设置为1，其他值设置为0
    # probabilities = np.where(np.isinf(probabilities), 1.0, 0.0)
    # 如果probabilities都为0，则取(0.5, 0.5)
    if np.sum(probabilities) == 0:
        p_list = [0.5, 0.5]
    else:
        p_list = probabilities / np.sum(probabilities)

    # 偏好加入总观众数
    preference_join_total = all_audience - random_join_total

    # 计算每个社区的加入观众数
    preference_join_community1 = int(preference_join_total * p_list[0])
    preference_join_community2 = preference_join_total - preference_join_community1
    # 更新社区1的观众数
    increment_distribution[1] += preference_join_community1
    # 更新社区2的观众数
    increment_distribution[2] += preference_join_community2

    return increment_distribution



# Function to calculate Mean Absolute Error (MAE)
def calculate_mae(vector1, vector2):
    return np.mean(np.abs(np.array(vector1) - np.array(vector2)))


# Function to evaluate and optimize parameters
def evaluate_and_optimize_parameters(p, q, t, train_data_t):
    mae_list = []

    for index, row in train_data_t.iterrows():
        true_distribution = row[DESIRED_COLUMNS1]
        predicted_distribution = simulate_audience_increment(row[DESIRED_COLUMNS], Nt_change, t, p, q)

        # Calculate Mean Absolute Error
        mae = calculate_mae(predicted_distribution, true_distribution)
        mae_list.append(mae)

    # Calculate the average MAE
    average_mae = np.mean(mae_list)

    return average_mae


# Function to calculate the best parameters for each time step
def calculate_best_params_for_time(t):
    best_params = None
    best_mae = float('inf')

    train_data_t = train_datasets[t]

    for p, q in product(PARAM_VALUES_P, PARAM_VALUES_Q):
        mae = evaluate_and_optimize_parameters(p, q, t, train_data_t)

        logging.info(f"Time Step {t}: Parameters: ({p}, {q}), MAE: {mae:.6f}")

        if mae < best_mae:
            best_params = (p, q)
            best_mae = mae

    logging.info(
        f"Time Step {t}: Best Parameters: {best_params}, MAE: {best_mae:.6f}")

    return t, best_params, best_mae


if __name__ == '__main__':
    results = []
    for t in tqdm(range(0,MAX_TIME_POINT), desc="Time Steps"):
        result = calculate_best_params_for_time(t)
        results.append(result)

    best_params_by_time = {}
    distances_by_time = {}

    # 获取每个时间步的最佳参数
    for t, best_params, best_mae in results:
        best_params_by_time[t] = best_params

        # 计算测试集上的平均MAE
        average_mae = evaluate_and_optimize_parameters(best_params[0], best_params[1], t, test_datasets[t])
        distances_by_time[t] = {'Average MAE': average_mae}

    # 创建DataFrame并保存为CSV
    df = pd.DataFrame({
        'Time Step': list(best_params_by_time.keys()),
        'Best Parameters': list(best_params_by_time.values()),
        'Average MAE': [distances_by_time[t]['Average MAE'] for t in best_params_by_time.keys()]
    })

    df.to_csv('./实验结果/params_and_distances_r=50_l1_test.csv', index=False)















# 十折交叉验证
# # Set numpy error handling
# np.seterr(divide='ignore', invalid='ignore')
#
# # Logging configuration
# logging.basicConfig(filename='./实验结果/pq_r=50_l0_true.log', level=logging.INFO, format='%(asctime)s - %(message)s')
#
# # Constants
#
# # LIVE_DATA_PATH = './data/live_data_distribution.csv'
# # CHANGE_DATA_PATH = './data/Nt_change.csv'
#
# LIVE_DATA_PATH = 'D:/研究生/毕业论文/代码/网络构建/直播平台网络模型/jupyter/处理数据/processed_hourly_stats_level_0.csv'
# CHANGE_DATA_PATH = 'D:/研究生/毕业论文/代码/网络构建/直播平台网络模型/jupyter/处理数据/change_rate_real_time_user_level_0.csv'
#
# DESIRED_COLUMNS = ['real_time_user', 'leave_user', 'user_count']
# DESIRED_COLUMNS1 = [f'{col}_t' for col in DESIRED_COLUMNS]
# MAX_TIME_POINT = 35
# PARAM_VALUES_P = np.arange(0, 1.1, step=0.1).round(1)
# PARAM_VALUES_Q = np.arange(-2, 2.1, step=0.5).round(1)
# RANDOM_SEED = 50
#
# # Load data
# live_data_distribution = pd.read_csv(LIVE_DATA_PATH, index_col=0)
# live_data_distribution = live_data_distribution.astype(int)
# Nt_change = pd.read_csv(CHANGE_DATA_PATH, index_col=0)
#
# # Generate time points
# time_points = np.arange(0, MAX_TIME_POINT)
#
#
# # Function to simulate audience increment
# def simulate_audience_increment(distribution_x, change_data, t, p, q):
#
#     increment_total_user = change_data.iloc[t, 0]#iloc使用整数索引，而不是列名
#     total_change = int(increment_total_user * (1+increment_total_user))
#     # total_change = int(distribution_x.iloc[0] * (1 + increment_total_user))
#     all_audience = int(total_change + distribution_x.iloc[2])
#
#     # 初始化增量分布
#     increment_distribution = [0] * len(distribution_x)
#     increment_distribution[0] = total_change
#
#     # 随机加入总观众数
#     random_join_total = int(all_audience * p)
#     # 计算每个社区的加入观众数
#     random_join_community1 = random_join_total * 0.5
#     random_join_community2 = random_join_total - random_join_community1
#     # 更新社区1的观众数
#     increment_distribution[1] += random_join_community1
#     # 更新社区2的观众数
#     increment_distribution[2] += random_join_community2
#
#     probabilities = np.array([distribution_x.iloc[i] ** q for i in range(1, 3)])
#     # # 将inf对应的概率设置为1，其他值设置为0
#     # probabilities = np.where(np.isinf(probabilities), 1.0, 0.0)
#     # 如果probabilities都为0，则取(0.5, 0.5)
#     if np.sum(probabilities) == 0:
#         p_list = [0.5, 0.5]
#     else:
#         p_list = probabilities / np.sum(probabilities)
#
#     # 偏好加入总观众数
#     preference_join_total = all_audience - random_join_total
#
#     # 计算每个社区的加入观众数
#     preference_join_community1 = int(preference_join_total * p_list[0])
#     preference_join_community2 = preference_join_total - preference_join_community1
#     # 更新社区1的观众数
#     increment_distribution[1] += preference_join_community1
#     # 更新社区2的观众数
#     increment_distribution[2] += preference_join_community2
#
#     return increment_distribution
#
#
# # Function to calculate Mean Absolute Error (MAE)
# def calculate_mae(vector1, vector2):
#     return np.mean(np.abs(np.array(vector1) - np.array(vector2)))
#
#
# # Function to evaluate and optimize parameters
# # 使用 K 折交叉验证
# def evaluate_and_optimize_parameters_cross_validation(p, q, t, data_t):
#     mae_list_cv = []
#     mae_list_test = []
#     for train_index, test_index in kf.split(data_t):
#         mae_cv = []
#         mae_test = []
#         train_data = data_t.iloc[train_index]
#         test_data = data_t.iloc[test_index]
#
#         # 计算交叉验证集上的 MAE
#         for index, row in train_data.iterrows():
#             true_distribution = row[DESIRED_COLUMNS1]
#             predicted_distribution = simulate_audience_increment(row[DESIRED_COLUMNS], Nt_change, t, p, q)
#             mae = calculate_mae(predicted_distribution, true_distribution)
#             mae_cv.append(mae)
#         average_cv = np.mean(mae_cv)
#         mae_list_cv.append(average_cv)
#
#
#         # 在测试集上计算 MAE
#         for index, row in test_data.iterrows():
#             true_distribution = row[DESIRED_COLUMNS1]
#             predicted_distribution = simulate_audience_increment(row[DESIRED_COLUMNS], Nt_change, t, p, q)
#             mae = calculate_mae(predicted_distribution, true_distribution)
#             mae_test.append(mae)
#         average_test = np.mean(mae_test)
#         mae_list_test.append(average_test)
#
#     # 计算交叉验证集和测试集上的平均 MAE
#     average_mae_cv = np.mean(mae_list_cv)
#     average_mae_test = np.mean(mae_list_test)
#
#     return average_mae_cv, average_mae_test
#
# # Function to calculate best parameters for each time step
# def calculate_best_params_for_time(t):
#     best_params = None
#     best_mae_cv = float('inf')
#
#     data_t = live_data_distribution[live_data_distribution.index == t]
#
#     for p, q in product(PARAM_VALUES_P, PARAM_VALUES_Q):
#         mae_cv, mae_test = evaluate_and_optimize_parameters_cross_validation(p, q, t, data_t)
#
#         logging.info(f"Time Step {t}: Parameters: ({p}, {q}), CV MAE: {mae_cv:.6f}, Test MAE: {mae_test:.6f}")
#
#         if mae_cv < best_mae_cv:
#             best_params = (p, q)
#             best_mae_cv = mae_cv
#             best_mae_test = mae_test
#
#     logging.info(f"Time Step {t}: Best Parameters: {best_params}, CV MAE: {best_mae_cv:.6f}, Test MAE: {best_mae_test:.6f}")
#
#     return t, best_params, best_mae_cv, best_mae_test
#
#
# if __name__ == '__main__':
#     # 定义K折交叉验证的折数
#     K = 10
#     kf = KFold(n_splits=K, shuffle=True, random_state=RANDOM_SEED)
#     results = []
#
#     for t in tqdm(range(0, MAX_TIME_POINT), desc="Time Steps"):
#         result = calculate_best_params_for_time(t)
#         results.append(result)
#
#     best_params_by_time = {}
#     distances_by_time = {}
#
#     # 获取每个时间步的最佳参数
#     for t, best_params, best_mae_cv, best_mae_test in results:
#         best_params_by_time[t] = best_params
#         distances_by_time[t] = {'CV MAE': best_mae_cv, 'Test MAE': best_mae_test}
#
#     # 创建 DataFrame 并保存为 CSV
#     df = pd.DataFrame({
#         'Time Step': list(best_params_by_time.keys()),
#         'Best Parameters': list(best_params_by_time.values()),
#         'CV MAE': [distances_by_time[t]['CV MAE'] for t in best_params_by_time.keys()],
#         'Test MAE': [distances_by_time[t]['Test MAE'] for t in best_params_by_time.keys()]
#     })
#
#     df.to_csv('./实验结果/params_and_distances_r=50_l0_true.csv', index=False)

