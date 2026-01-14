import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# Constants
LIVE_DATA_PATH = 'D:/研究生/论文/毕业论文/代码/网络构建/直播平台网络模型/jupyter/处理数据/processed_hourly_stats_level_1.csv'
CHANGE_DATA_PATH = 'D:/研究生/论文/毕业论文/代码/网络构建/直播平台网络模型/jupyter/处理数据/change_rate_real_time_user_level_1.csv'

DESIRED_COLUMNS = ['real_time_user', 'leave_user', 'user_count']
DESIRED_COLUMNS1 = [f'{col}_t' for col in DESIRED_COLUMNS]
MAX_TIME_POINT = 35
RANDOM_SEED = 50

def calculate_mae(vector1, vector2):
    return np.mean(np.abs(np.array(vector1) - np.array(vector2)))

live_data_distribution = pd.read_csv(LIVE_DATA_PATH, index_col=0)
live_data_distribution = live_data_distribution.astype(int)

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

# Function to train and evaluate model
def train_and_evaluate_model(model, train_data, test_data):

    X_train = train_data[DESIRED_COLUMNS].values
    y_train = train_data[DESIRED_COLUMNS1].values
    X_test = test_data[DESIRED_COLUMNS].values
    y_test = test_data[DESIRED_COLUMNS1].values

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae_list = []
    for i in range(y_test.shape[0]):
        mae = calculate_mae(y_pred[i], y_test[i])
        mae_list.append(mae)
    average_mae = np.mean(mae_list)

    return average_mae

def train_and_evaluate_model_svr(model, train_data, test_data):
    X_train = train_data[DESIRED_COLUMNS].values
    y_train = train_data[DESIRED_COLUMNS1].values
    X_test = test_data[DESIRED_COLUMNS].values
    y_test = test_data[DESIRED_COLUMNS1].values

    mae_list = []
    for i in range(y_train.shape[1]):  # Loop through each target variable
        model.fit(X_train, y_train[:, i])  # Train on one target variable at a time
        y_pred = model.predict(X_test)
        mae = calculate_mae(y_pred, y_test[:, i])  # Calculate MAE for this target
        mae_list.append(mae)

    average_mae = np.mean(mae_list)  # Average MAE across all target variables
    return average_mae

# Initialize models
#思考其他对比实验
dt_model = DecisionTreeRegressor(random_state=RANDOM_SEED)
svr_model = SVR()

results_dt = []
results_svr = []

for t in range(MAX_TIME_POINT):

    train_data_t = train_datasets[t]
    test_data_t = test_datasets[t]

    # if len(train_data_t) > 1000:
    #     # 从 train_data 数据框中随机采样 1000行数据
    #     train_data_t = train_data_t.sample(n=1000, random_state=RANDOM_SEED)

    # Train and evaluate decision tree model
    dt_mae = train_and_evaluate_model(dt_model, train_data_t, test_data_t)
    results_dt.append((t, dt_mae))

    # Train and evaluate Support Vector Regressor model
    svr_mae = train_and_evaluate_model_svr(svr_model, train_data_t, test_data_t)
    results_svr.append((t, svr_mae))


df_dt = pd.DataFrame(results_dt, columns=['Time Step', 'MAE'])
# df_dt.to_csv('./对比实验结果/results_dt_50_db_level0.csv', index=False, float_format='%.6f')

df_svr = pd.DataFrame(results_svr, columns=['Time Step', 'MAE'])
df_svr.to_csv('./对比实验结果/results_svr_50_db_level1.csv', index=False, float_format='%.6f')
