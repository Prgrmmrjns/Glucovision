import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings
import json
import optuna
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from params import *
from processing_functions import *

warnings.filterwarnings('ignore')

# Script-specific constants
optimization_features_baseline = ['insulin']  # Baseline uses only insulin
fast_features = ['insulin']
load_params = False
n_trials = 300
random_seed = 42
n_jobs = -1
current_patient_weight = 10
validation_size = 0.2
approach = 'baseline'

# Functions imported from processing_functions.py

def get_data(patient):
    glucose_data = pd.read_csv(f"../diabetes_subset_pictures-glucose-food-insulin/{patient}/glucose.csv")
    insulin_data = pd.read_csv(f"../diabetes_subset_pictures-glucose-food-insulin/{patient}/insulin.csv")
    glucose_data["datetime"] = pd.to_datetime(glucose_data["date"] + ' ' + glucose_data["time"])
    glucose_data = glucose_data.drop(['type', 'comments', 'date', 'time'], axis=1)
    glucose_data['glucose'] *= 18.0182
    glucose_data['hour'] = glucose_data['datetime'].dt.hour
    glucose_data['time'] = glucose_data['hour'] + glucose_data['datetime'].dt.minute / 60
    insulin_data["datetime"] = pd.to_datetime(insulin_data["date"] + ' ' + insulin_data["time"])
    insulin_data.fillna(0, inplace=True)
    insulin_data['insulin'] = insulin_data['slow_insulin'] + insulin_data['fast_insulin']
    insulin_data = insulin_data.drop(['slow_insulin', 'fast_insulin', 'comment', 'date', 'time'], axis=1)
    combined_data = insulin_data
    for horizon in PREDICTION_HORIZONS:
        glucose_data[f'glucose_{horizon}'] = glucose_data['glucose'].shift(-horizon) - glucose_data['glucose']
    glucose_data['glucose_change'] = glucose_data['glucose'] - glucose_data['glucose'].shift(1)
    glucose_data['glucose_change_projected'] = glucose_data['glucose_change'].rolling(6, min_periods=6).apply(lambda window: get_projected_value(window, 6))
    glucose_data['glucose_projected'] = glucose_data['glucose'].rolling(6, min_periods=6).apply(lambda window: get_projected_value(window, 6))
    glucose_data.dropna(subset=[f'glucose_24'], inplace=True)
    glucose_data['patient_id'] = patient
    return glucose_data, combined_data

def add_features(params, features, data):
    glucose_data, combined_data = data
    glucose_data = glucose_data.copy()
    time_diff_hours = (glucose_data['datetime'].values.astype(np.int64)[:, None] - combined_data['datetime'].values.astype(np.int64)[None, :]) / 3600000000000
    base_curves = {f: bezier_curve(np.array(params[f]).reshape(-1, 2), num=50) for f in features if f in params}
    for i, feature in enumerate(features):
        curve = base_curves[feature]
        x_curve, y_curve = curve[:, 0], curve[:, 1]
        valid_mask = (time_diff_hours >= 0) & (time_diff_hours <= x_curve[-1])
        weights = np.zeros_like(time_diff_hours)
        weights[valid_mask] = y_curve[np.clip(np.searchsorted(x_curve, time_diff_hours[valid_mask]), 0, len(y_curve) - 1)]
        feature_values = combined_data[feature].values
        glucose_data[feature] = np.dot(weights, feature_values)
    return glucose_data

def objective(trial):
    params = {}
    max_x_values = np.where(np.isin(optimization_features_baseline, fast_features), 3.0, 8.0)
    for i, f in enumerate(optimization_features_baseline):
        params[f] = [
            0.0, 0.0,
            trial.suggest_float(f"{f}_x2", 0.0, max_x_values[i]), trial.suggest_float(f"{f}_y2", 0.0, 1.0),
            trial.suggest_float(f"{f}_x3", 0.0, max_x_values[i]), trial.suggest_float(f"{f}_y3", 0.0, 1.0),
            trial.suggest_float(f"{f}_x4", 0.0, max_x_values[i]), trial.suggest_float(f"{f}_y4", 0.0, 1.0),
            trial.suggest_float(f"{f}_x5", 0.0, max_x_values[i]), 0,
        ]
    
    patient_data_list = [add_features(params, optimization_features_baseline, data) for data in all_train_data]
    X_all = pd.concat([data[data.columns.difference(features_to_remove_ph)] for data in patient_data_list], ignore_index=True)
    y_all = pd.concat([data['glucose_12'] for data in patient_data_list], ignore_index=True)
    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=validation_size, random_state=42)
    model = lgb.train(LGB_PARAMS, lgb.Dataset(X_train, label=y_train), valid_sets=[lgb.Dataset(X_val, label=y_val)])
    return np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
    
if load_params:
    all_data = pd.read_csv(f'../results/{approach}_data.csv')
    
else:
    features_to_remove_ph = FEATURES_TO_REMOVE_D1NAMO + [f'glucose_{h}' for h in PREDICTION_HORIZONS]
    all_train_data = []
    for patient in PATIENTS_D1NAMO:
        glucose_data, combined_data = get_data(patient)
        train_days = glucose_data['datetime'].dt.day.unique()[:3]
        train_glucose = glucose_data[glucose_data['datetime'].dt.day.isin(train_days)]
        train_combined = combined_data[combined_data['datetime'].dt.day.isin(train_days)]
        all_train_data.append((train_glucose, train_combined))
    
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)
    print(f"Combined optimization best score: {study.best_value:.4f}")
    
    # Save globally optimized Bezier parameters
    global_params = {f: [0.0, 0.0,
                        study.best_params[f"{f}_x2"], study.best_params[f"{f}_y2"], 
                        study.best_params[f"{f}_x3"], study.best_params[f"{f}_y3"],
                        study.best_params[f"{f}_x4"], study.best_params[f"{f}_y4"],
                        study.best_params[f"{f}_x5"], 0] for f in optimization_features_baseline}
    
    # Store global parameters - same for all patients since we use global optimization
    json.dump(global_params, open(f'../results/{approach}_bezier_params.json', 'w'), indent=2)
    all_data_list = []
    for p in PATIENTS_D1NAMO:
        patient_data = add_features(global_params, optimization_features_baseline, get_data(p))
        patient_data['patient_id'] = f"patient_{p}"
        all_data_list.append(patient_data)
    all_data = pd.concat(all_data_list, ignore_index=True)
    all_data.to_csv(f'../results/{approach}_data.csv', index=False)

# Evaluation phase
all_results = []
for prediction_horizon in PREDICTION_HORIZONS:
    target_feature = f'glucose_{prediction_horizon}'
    features_to_remove_ph = FEATURES_TO_REMOVE_D1NAMO + [f'glucose_{h}' for h in PREDICTION_HORIZONS]
    results_list = []
    for patient in PATIENTS_D1NAMO:
        patient_mask = all_data['patient_id'] == f"patient_{patient}"
        days = all_data[patient_mask]['datetime'].dt.day.unique()[3:]
        for test_day in days:
            day_hours = all_data[patient_mask & (all_data['datetime'].dt.day == test_day)]['hour'].unique()
            for hour in day_hours:
                test = all_data[patient_mask & (all_data['datetime'].dt.day == test_day) & (all_data['hour'] == hour)]
                X = pd.concat([all_data[patient_mask & (all_data['datetime'].shift(-6) < test['datetime'].min())], all_data[~patient_mask]])
                indices = train_test_split(range(len(X)), test_size=validation_size, random_state=42)
                weights = [np.where(X['patient_id'].values[idx] == f"patient_{patient}", current_patient_weight, 1) for idx in indices]
                available_features = X.columns.difference(features_to_remove_ph)
                train = X[available_features]
                X_train, y_train, weights_train = train.values[indices[0]], X[target_feature].values[indices[0]], weights[0]
                X_val, y_val, weights_val = train.values[indices[1]], X[target_feature].values[indices[1]], weights[1]
                model = lgb.train(LGB_PARAMS, lgb.Dataset(X_train, label=y_train, weight=weights_train), 
                                valid_sets=[lgb.Dataset(X_val, label=y_val, weight=weights_val)])
                test_features = test[available_features]
                predictions = model.predict(test_features.values)
                ground_truth = test[target_feature].values
                rmse = np.sqrt(mean_squared_error(ground_truth, predictions))
                results_list.append([prediction_horizon, patient, test_day, hour, rmse])
    df = pd.DataFrame(results_list, columns=['Prediction Horizon', 'Patient', 'Day', 'Hour', 'RMSE'])
    print(f"Average for prediction horizon {prediction_horizon}: {df['RMSE'].mean():.4f}")
    all_results.extend(results_list)

df = pd.DataFrame(all_results, columns=['Prediction Horizon', 'Patient', 'Day', 'Hour', 'RMSE'])
df.to_csv(f'../results/{approach}_predictions.csv', index=False)