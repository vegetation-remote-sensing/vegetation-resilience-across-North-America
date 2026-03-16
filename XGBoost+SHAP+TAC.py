# -*- coding: utf-8 -*-
import gc
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
import shap
from sklearn.model_selection import RandomizedSearchCV

# path
input_path = "./input/"
output_path = "./output/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# parameters
param_dist = {
    'n_estimators': np.arange(50, 1000, 50), #1000
    'max_depth': np.arange(3, 10), #9
    'gamma': np.linspace(0.0001, 0.0030, 30), #0.0026
    'min_child_weight': np.arange(1, 10), #7
    'subsample': np.linspace(0.5, 1.0, 5), #0.9
    'colsample_bytree': np.linspace(0.5, 1.0, 5), #0.5
    'learning_rate': np.linspace(0.01, 0.20, 20), #0.04
    'reg_alpha': np.linspace(-10, 10, 21), #0
    'reg_lambda': np.linspace(-10, 10, 21), #10
}

# load data
data_file = input_path + "xgboost_data.csv"
data = pd.read_csv(data_file)
x_columns = ['FC', 'FL', 'FG', 'IF', 'RF', 'PF',
             'CH', 'TD', 'FA', 'LNC', 'LPC', 'SLA', 'LDMC', 'SD', 'NSR', 'RD',
             'PDSI', 'PDSI_Sen', 'PDSI_CV', 'PRE', 'PRE_Sen', 'PRE_CV', 'TMP', 'TMP_Sen', 'TMP_CV',
             'Tmax', 'Tmax_Sen', 'Tmax_CV', 'Tmin', 'Tmin_Sen', 'Tmin_CV', 'Solar', 'Solar_Sen', 'Solar_CV',
             'LST', 'LST_Sen', 'LST_CV',  'WTD', 'VPD', 'VPD_Sen', 'VPD_CV',
             'SM', 'SM_Sen', 'SM_CV', 'CEC', 'BDOD', 'Clay', 'Sand', 'PH',  'TN',  'SOC',  'OCD',  'AWC',
             'HFP',  'HFP_Sen',  'NTL',  'NTL_Sen',  'Fire',
             'ELEV',  'SLP',  'ASP',  'TopoDV']  # drivers
y_column = 'AR1_kNDVI' #target

x_all = data[x_columns].values
y_all = data[y_column].values

# XGBoost-RandomizedSearchCV
xgb_model = xgb.XGBRegressor(random_state=483)

random_search = RandomizedSearchCV(xgb_model, param_distributions=param_dist, n_iter=50, scoring='neg_mean_squared_error', n_jobs=40, verbose=1, random_state=483)
random_search.fit(x_all, y_all)

# best parameters
print("Best parameters found: ", random_search.best_params_)

# predict with best parameters
best_model = random_search.best_estimator_
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=483)

# 训练与评估
best_model.fit(x_train, y_train)
train_pred = best_model.predict(x_train)
test_pred = best_model.predict(x_test)

train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

print(f"Train RMSE: {train_rmse}, Test RMSE: {test_rmse}")
print(f"Train R2: {train_r2}, Test R2: {test_r2}")

# 使用SHAP解释模型
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(x_all)

# # 输出SHAP值的摘要图
# shap.summary_plot(shap_values, x_train, feature_names=x_columns)

# 保存SHAP的结果
shap_values_df = pd.DataFrame(shap_values, columns=x_columns)
shap_values_df.to_csv(output_path+"shap_values_TAC.csv", index=False)

# 可视化特征重要性
shap_importance = np.abs(shap_values).mean(0)
shap_importance_df = pd.DataFrame({"Feature": x_columns, "SHAP Importance": shap_importance})
shap_importance_df = shap_importance_df.sort_values(by="SHAP Importance", ascending=False)

# 保存SHAP特征重要性
shap_importance_df.to_csv(output_path+"shap_importance_TAC.csv", index=False)

# 显示前10个最重要特征
print(shap_importance_df.head(10))


