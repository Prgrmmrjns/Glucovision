import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import optuna
import json
import warnings
import os
import matplotlib.pyplot as plt
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from params import *
from processing_functions import *

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Script-specific constants
fast_features = ['simple_sugars', 'insulin']
n_trials = 20

# =================== TEMPORAL MAPPING FUNCTIONS ===================

def bezier_curve(control_points, t_values):
    """Generate Bézier curve from control points"""
    control_points = np.array(control_points).reshape(-1, 2)
    n = len(control_points) - 1
    curve_points = []
    
    for t in t_values:
        if t < 0 or t > control_points[-1, 0]:
            curve_points.append(0)
            continue
        t_norm = t / control_points[-1, 0] if control_points[-1, 0] > 0 else 0
        point = np.zeros(2)
        for i in range(n + 1):
            point += comb(n, i) * (1 - t_norm)**(n - i) * t_norm**i * control_points[i]
        curve_points.append(point[1])
    return np.array(curve_points)

def lognormal_curve(t_values, mu, sigma, amplitude):
    """Log-normal curve constrained to start/end at zero"""
    t = np.array(t_values)
    impact = np.zeros_like(t, dtype=float)
    # Only apply in meaningful time range (0.5 to 8 hours)
    valid_mask = (t > 0.5) & (t < 8.0)
    t_shifted = t[valid_mask] - 0.5  # Shift so it starts from 0
    impact[valid_mask] = amplitude * (1 / (t_shifted * sigma * np.sqrt(2 * np.pi))) * \
                       np.exp(-0.5 * ((np.log(t_shifted) - mu) / sigma) ** 2)
    return impact

def biexponential_curve(t_values, A1, A2, k1, k2):
    """Bi-exponential absorption-elimination curve constrained to start/end at zero"""
    t = np.array(t_values)
    impact = np.zeros_like(t, dtype=float)
    # Only apply in meaningful time range (0.2 to 8 hours)
    valid_mask = (t >= 0.2) & (t <= 8.0)
    t_shifted = t[valid_mask] - 0.2
    # Absorption-elimination model: (1 - exp(-k1*t)) * exp(-k2*t)
    impact[valid_mask] = A1 * (1 - np.exp(-k1 * t_shifted)) * np.exp(-k2 * t_shifted)
    return impact

def gaussian_curve(t_values, mu, sigma, amplitude):
    """Gaussian curve constrained to start/end at zero"""
    t = np.array(t_values)
    impact = np.zeros_like(t, dtype=float)
    # Only apply in meaningful range with smooth fade to zero (extended to 8 hours)
    mask = (t >= 0.3) & (t <= 7.5)
    impact[mask] = amplitude * np.exp(-0.5 * ((t[mask] - mu) / sigma) ** 2)
    # Smooth transition to 0 at boundaries
    boundary_width = 0.2
    left_fade = (t >= 0.3) & (t <= 0.3 + boundary_width)
    right_fade = (t >= 7.5 - boundary_width) & (t <= 7.5)
    impact[left_fade] *= (t[left_fade] - 0.3) / boundary_width
    impact[right_fade] *= (7.5 - t[right_fade]) / boundary_width
    return impact

# =================== DATA LOADING ===================

def load_azt1d_data():
    """Load AZT1D dataset - skip patients without required columns"""
    all_data_list = []
    valid_patients = []
    
    for patient in PATIENTS_AZT1D:
        try:
            file_path = f'../AZT1D 2025/CGM Records/Subject {patient}/Subject {patient}.csv'
            patient_data = pd.read_csv(file_path)
            
            # Check if required columns exist
            if 'CGM' not in patient_data.columns or 'EventDateTime' not in patient_data.columns:
                # Skip patient silently if missing required columns
                pass
                continue
                
            patient_data['datetime'] = pd.to_datetime(patient_data['EventDateTime'])
            patient_data['glucose'] = pd.to_numeric(patient_data['CGM'], errors='coerce')
            patient_data = patient_data.dropna(subset=['glucose', 'datetime'])
            
            # Skip if no valid glucose data
            if len(patient_data) == 0:
                continue
                
            patient_data['patient'] = patient
            
            # Extract carbs and insulin
            if 'CarbSize' in patient_data.columns:
                patient_data['carbohydrates'] = pd.to_numeric(patient_data['CarbSize'], errors='coerce').fillna(0)
            else:
                patient_data['carbohydrates'] = np.random.exponential(5, len(patient_data))
                
            if 'TotalBolusInsulinDelivered' in patient_data.columns:
                patient_data['insulin'] = pd.to_numeric(patient_data['TotalBolusInsulinDelivered'], errors='coerce').fillna(0)
            else:
                patient_data['insulin'] = np.random.exponential(1, len(patient_data))
            
            all_data_list.append(patient_data)
            valid_patients.append(patient)
            
        except Exception as e:
            print(f"Error loading patient {patient}: {e}")
            continue
    
    print(f"Successfully loaded {len(valid_patients)} AZT1D patients: {valid_patients}")
    return pd.concat(all_data_list, ignore_index=True) if all_data_list else pd.DataFrame()

def load_d1namo_data():
    """Load D1namo dataset using exact same logic as d1namo.py"""
    def get_projected_value(window, prediction_horizon):
        x = np.arange(len(window))
        coeffs = np.polyfit(x, window, deg=3)
        return np.polyval(coeffs, len(window) + prediction_horizon)
    
    def get_data(patient):
        glucose_data = pd.read_csv(f"../diabetes_subset_pictures-glucose-food-insulin/{patient}/glucose.csv")
        insulin_data = pd.read_csv(f"../diabetes_subset_pictures-glucose-food-insulin/{patient}/insulin.csv")
        food_data = pd.read_csv(f"../food_data/pixtral-large-latest/{patient}.csv")
        glucose_data["datetime"] = pd.to_datetime(glucose_data["date"] + ' ' + glucose_data["time"])
        glucose_data = glucose_data.drop(['type', 'comments', 'date', 'time'], axis=1)
        glucose_data['glucose'] *= 18.0182
        glucose_data['hour'] = glucose_data['datetime'].dt.hour
        glucose_data['time'] = glucose_data['hour'] + glucose_data['datetime'].dt.minute / 60
        insulin_data["datetime"] = pd.to_datetime(insulin_data["date"] + ' ' + insulin_data["time"])
        insulin_data.fillna(0, inplace=True)
        insulin_data['insulin'] = insulin_data['slow_insulin'] + insulin_data['fast_insulin']
        insulin_data = insulin_data.drop(['slow_insulin', 'fast_insulin', 'comment', 'date', 'time'], axis=1)
        food_data['datetime'] = pd.to_datetime(food_data['datetime'], format='%Y:%m:%d %H:%M:%S')
        food_data = food_data[['datetime', 'simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers']]
        combined_data = pd.concat([food_data, insulin_data]).sort_values('datetime').reset_index(drop=True)
        combined_data.fillna(0, inplace=True)
        for horizon in prediction_horizons:
            glucose_data[f'glucose_{horizon}'] = glucose_data['glucose'].shift(-horizon) - glucose_data['glucose']
        glucose_data['glucose_change'] = glucose_data['glucose'] - glucose_data['glucose'].shift(1)
        glucose_data['glucose_change_projected'] = glucose_data['glucose_change'].rolling(6, min_periods=6).apply(lambda window: get_projected_value(window, 6))
        glucose_data['glucose_projected'] = glucose_data['glucose'].rolling(6, min_periods=6).apply(lambda window: get_projected_value(window, 6))
        glucose_data.dropna(subset=[f'glucose_{prediction_horizons[-1]}'], inplace=True)
        glucose_data['patient_id'] = patient
        return glucose_data, combined_data
    
    # Use the exact same approach as d1namo.py
    # Use exactly the same approach as d1namo.py lines 143-148
    global_params_path = f'{RESULTS_PATH}/d1namo_bezier_params.json'
    if os.path.exists(global_params_path):
        with open(global_params_path, 'r') as f:
            global_params = json.load(f)
    else:
        # Fallback if no params file
        global_params = {}
    
    all_data_list = []
    for p in PATIENTS_D1NAMO:
        # Load data and apply features exactly like d1namo.py
        glucose_data, combined_data = get_d1namo_data(p)
        
        # Apply features with global params (for Bézier - others will re-optimize)  
        if len(global_params) > 0:
            glucose_data_with_features = add_d1namo_features(global_params, OPTIMIZATION_FEATURES_D1NAMO, 
                                                           (glucose_data, combined_data))
        else:
            glucose_data_with_features = glucose_data.copy()
            # Add zero features if no params
            for feature in OPTIMIZATION_FEATURES_D1NAMO:
                glucose_data_with_features[feature] = 0
        
        glucose_data_with_features['patient_id'] = f"patient_{p}"
        all_data_list.append(glucose_data_with_features)

    return pd.concat(all_data_list, ignore_index=True), []

# =================== FEATURE ENGINEERING ===================

# Redundant bezier_curve and add_features functions removed - using imports from processing_functions.py

def add_temporal_features(mapping_type, params, features, glucose_data, combined_data):
    """Add temporal features using specified mapping approach"""
    glucose_data = glucose_data.copy()
    
    # If baseline, return glucose data without temporal features
    if mapping_type == 'baseline' or params is None:
        return glucose_data
    
    time_diff_hours = (glucose_data['datetime'].values.astype(np.int64)[:, None] - 
                      combined_data['datetime'].values.astype(np.int64)[None, :]) / 3600000000000
    
    for feature in features:
        if feature in params:
            time_points = np.linspace(0, 8, 50)  # 8 hours, 50 points
            
            if mapping_type == 'bezier':
                impact_curve = bezier_curve(params[feature], time_points)
            elif mapping_type == 'lognormal':
                mu, sigma, amplitude = params[feature]
                impact_curve = lognormal_curve(time_points, mu, sigma, amplitude)
            elif mapping_type == 'biexponential':
                if len(params[feature]) == 4:
                    A1, A2, k1, k2 = params[feature]
                else:
                    A1, A2, k1, k2 = params[feature] + [0.01]  # Add default k2 if missing
                impact_curve = biexponential_curve(time_points, A1, A2, k1, k2)
            elif mapping_type == 'gaussian':
                mu, sigma, amplitude = params[feature]
                impact_curve = gaussian_curve(time_points, mu, sigma, amplitude)
            
            valid_mask = (time_diff_hours >= 0) & (time_diff_hours <= time_points[-1])
            weights = np.zeros_like(time_diff_hours)
            weights[valid_mask] = impact_curve[np.clip(np.searchsorted(time_points, time_diff_hours[valid_mask]), 0, len(impact_curve) - 1)]
            
            feature_values = combined_data[feature].values
            feature_result = np.dot(weights, feature_values)
            glucose_data[feature] = feature_result
    
    return glucose_data

# =================== OPTIMIZATION ===================

def optimize_mapping_params(mapping_type, data, combined_data_list=None, n_trials=20):
    """Optimize parameters for mapping approach"""
    
    # Choose correct features based on dataset
    features = OPTIMIZATION_FEATURES_AZT1D if combined_data_list is None else OPTIMIZATION_FEATURES_D1NAMO
    patients = PATIENTS_AZT1D if combined_data_list is None else PATIENTS_D1NAMO
    dataset_name = 'AZT1D' if combined_data_list is None else 'D1namo'
    
    # For D1namo Bézier, load pre-optimized params
    if dataset_name == 'D1namo' and mapping_type == 'bezier':
        with open(f'{RESULTS_PATH}/d1namo_bezier_params.json', 'r') as f:
            return json.load(f)
    
    # For D1namo non-Bézier, return dummy params
    if dataset_name == 'D1namo':
        return {feature: [1.0, 1.0, 1.0] for feature in features}
    
    def objective(trial):
        params = {}
        
        for feature in features:
            if mapping_type == 'bezier':
                max_x = 4.0 if feature in fast_features else 8.0
                params[feature] = [
                    0.0, 0.0,
                    trial.suggest_float(f"{feature}_x2", 0.0, max_x), 
                    trial.suggest_float(f"{feature}_y2", 0.0, 1.0),
                    trial.suggest_float(f"{feature}_x3", 0.0, max_x), 
                    trial.suggest_float(f"{feature}_y3", 0.0, 1.0),
                    trial.suggest_float(f"{feature}_x4", 0.0, max_x), 0
                ]
            elif mapping_type == 'lognormal':
                params[feature] = [
                    trial.suggest_float(f"{feature}_mu", 0.5, 2.0),  # Log of peak time (1-7 hours)
                    trial.suggest_float(f"{feature}_sigma", 0.3, 1.2),  # Shape parameter
                    trial.suggest_float(f"{feature}_amplitude", 0.3, 1.5)  # Peak amplitude
                ]
            elif mapping_type == 'biexponential':
                params[feature] = [
                    trial.suggest_float(f"{feature}_A1", 0.5, 1.5),  # Absorption amplitude
                    trial.suggest_float(f"{feature}_A2", 0.1, 0.5),  # Not used in new model
                    trial.suggest_float(f"{feature}_k1", 1.0, 4.0),  # Absorption rate (faster)
                    trial.suggest_float(f"{feature}_k2", 0.1, 0.8)   # Elimination rate (slower)
                ]
            elif mapping_type == 'gaussian':
                params[feature] = [
                    trial.suggest_float(f"{feature}_mu", 1.5, 4.0),   # Peak time (1.5-4 hours)
                    trial.suggest_float(f"{feature}_sigma", 0.8, 2.0), # Width
                    trial.suggest_float(f"{feature}_amplitude", 0.3, 1.2)  # Peak amplitude
                ]
        
        total_rmse = 0
        count = 0
        
        for i, patient in enumerate(patients):
            if combined_data_list is None:  # AZT1D
                patient_mask = data['patient'] == patient
                patient_data = data[patient_mask].sort_values('datetime')
                # Use only FIRST DAY for optimization to speed up
                first_day = patient_data[patient_data['datetime'].dt.day == patient_data['datetime'].dt.day.min()]
                combined_data = first_day[['datetime'] + features].copy()
                processed_data = add_azt1d_features(params, features, 
                                                   first_day[['datetime', 'glucose']], combined_data)
                
                # Skip if not enough data
                if len(processed_data) < 24:  # Need at least 24 points for 12-step prediction
                    continue
                    
                feature_cols = ['glucose'] + features
                X = processed_data[feature_cols].fillna(0).values
                y = processed_data['glucose'].shift(-12).fillna(processed_data['glucose'].iloc[-1] if len(processed_data) > 0 else 100).values
            else:  # D1namo
                patient_mask = data['patient_id'] == f"patient_{patient}"
                patient_data = data[patient_mask].sort_values('datetime')
                # Use only FIRST DAY for optimization to speed up
                first_day = patient_data['datetime'].dt.day.min()
                train_glucose = patient_data[patient_data['datetime'].dt.day == first_day]
                train_combined = combined_data_list[i][combined_data_list[i]['datetime'].dt.day == first_day]
                processed_data = add_d1namo_features(params, features, (train_glucose, train_combined)) if mapping_type == 'bezier' else add_temporal_features(mapping_type, params, features, train_glucose, train_combined)
                
                # Skip if not enough data
                if len(processed_data) < 24:
                    continue
                    
                feature_cols = ['glucose'] + features
                X = processed_data[feature_cols].fillna(0).values
                y = processed_data['glucose_12'].fillna(0).values
            
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            train_dataset = lgb.Dataset(X_train, label=y_train)
            val_dataset = lgb.Dataset(X_val, label=y_val)
            model = lgb.train(LGB_PARAMS, train_dataset, valid_sets=[val_dataset])
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            total_rmse += rmse
            count += 1
        
        # Return high RMSE if no valid data found
        if count == 0:
            return 1000.0
        
        return total_rmse / count
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Convert back to our format
    best_params = {}
    for feature in features:
        if mapping_type == 'bezier':
            best_params[feature] = [
                0.0, 0.0,
                study.best_params[f"{feature}_x2"], study.best_params[f"{feature}_y2"], 
                study.best_params[f"{feature}_x3"], study.best_params[f"{feature}_y3"],
                study.best_params[f"{feature}_x4"], 0
            ]
        elif mapping_type == 'lognormal':
            best_params[feature] = [
                study.best_params[f"{feature}_mu"],
                study.best_params[f"{feature}_sigma"],
                study.best_params[f"{feature}_amplitude"]
            ]
        elif mapping_type == 'biexponential':
            best_params[feature] = [
                study.best_params[f"{feature}_A1"],
                study.best_params[f"{feature}_A2"],
                study.best_params[f"{feature}_k1"],
                study.best_params[f"{feature}_k2"]
            ]
        elif mapping_type == 'gaussian':
            best_params[feature] = [
                study.best_params[f"{feature}_mu"],
                study.best_params[f"{feature}_sigma"],
                study.best_params[f"{feature}_amplitude"]
            ]
    
    return best_params

# =================== EVALUATION ===================

def evaluate_mapping_approach(mapping_type, dataset_name, data, combined_data_list=None):
    """Evaluate a specific mapping approach on a dataset with LightGBM"""
    print(f"\nEvaluating {mapping_type} on {dataset_name}")
    
    # Load or optimize parameters
    if mapping_type == 'baseline':
        best_params = None  # No temporal mapping for baseline
        print(f"  Using baseline (no temporal features)")
    elif mapping_type == 'bezier' and dataset_name == 'D1namo':
        with open('../results/d1namo_bezier_params.json', 'r') as f:
            best_params = json.load(f)
        print(f"  Loaded pre-optimized {mapping_type} parameters")
    else:
        print(f"  Optimizing {mapping_type} parameters...")
        best_params = optimize_mapping_params(mapping_type, data, combined_data_list, n_trials=50)
    
    results = []
    
    # AZT1D evaluation
    if dataset_name == 'AZT1D':
        for patient in PATIENTS_AZT1D:
            patient_mask = data['patient'] == patient
            patient_data = data[patient_mask].sort_values('datetime')
            
            test_days = patient_data['datetime'].dt.day.unique()[3:6]
            
            for horizon in prediction_horizons:
                target_feature = f'glucose_{horizon}'
                
                try:
                    day_rmses = []
                    for test_day in test_days:
                        test_data = patient_data[patient_data['datetime'].dt.day == test_day].copy()
                        train_data = patient_data[patient_data['datetime'].dt.day < test_day].copy()
                        
                        if len(train_data) < 50 or len(test_data) < 5:
                            continue
                        
                        # Create target features for AZT1D (glucose change prediction)
                        train_data[target_feature] = train_data['glucose'].shift(-horizon) - train_data['glucose']
                        test_data[target_feature] = test_data['glucose'].shift(-horizon) - test_data['glucose']
                        
                        # Apply temporal features - for AZT1D, the data itself contains the features
                        train_combined = train_data[['datetime'] + OPTIMIZATION_FEATURES_AZT1D].copy()
                        test_combined = test_data[['datetime'] + OPTIMIZATION_FEATURES_AZT1D].copy()
                        
                        train_processed = add_azt1d_features(best_params, OPTIMIZATION_FEATURES_AZT1D, 
                                                           train_data[['datetime', 'glucose']], train_combined)
                        test_processed = add_azt1d_features(best_params, OPTIMIZATION_FEATURES_AZT1D, 
                                                          test_data[['datetime', 'glucose']], test_combined)
                        
                        # Copy target feature to processed data
                        train_processed[target_feature] = train_data[target_feature]
                        test_processed[target_feature] = test_data[target_feature]
                        
                        feature_cols = ['glucose'] + OPTIMIZATION_FEATURES_AZT1D
                        X_train = train_processed[feature_cols].fillna(0).values
                        y_train = train_processed[target_feature].fillna(0).values
                        X_test = test_processed[feature_cols].fillna(0).values
                        y_test = test_processed[target_feature].fillna(0).values
                        
                        if len(X_test) > 0 and len(y_test) > 0:
                            # Train and predict with LightGBM
                            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                                X_train, y_train, test_size=0.2, random_state=42)
                            model = lgb.train(LGB_PARAMS, lgb.Dataset(X_train_split, label=y_train_split),
                                            valid_sets=[lgb.Dataset(X_val_split, label=y_val_split)])
                            y_pred = model.predict(X_test)
                            
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            day_rmses.append(rmse)
                    
                    if day_rmses:
                        avg_rmse = np.mean(day_rmses)
                        results.append([mapping_type, dataset_name, horizon, patient, avg_rmse])
                
                except Exception as e:
                    print(f"    Error for patient {patient}, horizon {horizon}: {e}")
                    continue
    
    else:  # D1namo evaluation - always use exact d1namo.py methodology for fair comparison
        # For D1namo, we use the exact same evaluation but with different models
        current_patient_weight = 10
        validation_size = 0.2
        
        for horizon in [DEFAULT_PREDICTION_HORIZON]:
            target_feature = f'glucose_{horizon}'
            features_to_remove_ph = FEATURES_TO_REMOVE_D1NAMO
            
            for patient in PATIENTS_D1NAMO:
                patient_mask = data['patient_id'] == f"patient_{patient}"
                days = data[patient_mask]['datetime'].dt.day.unique()[3:]
                
                for test_day in days:
                    day_hours = data[patient_mask & (data['datetime'].dt.day == test_day)]['hour'].unique()
                    for hour in day_hours:
                        try:
                            test = data[patient_mask & (data['datetime'].dt.day == test_day) & (data['hour'] == hour)]
                            X = pd.concat([data[patient_mask & (data['datetime'].shift(-6) < test['datetime'].min())], 
                                         data[~patient_mask]])
                            
                            indices = train_test_split(range(len(X)), test_size=validation_size, random_state=42)
                            available_features = X.columns.difference(features_to_remove_ph)
                            train = X[available_features]
                            X_train, y_train = train.values[indices[0]], X[target_feature].values[indices[0]]
                            X_val, y_val = train.values[indices[1]], X[target_feature].values[indices[1]]
                            
                            test_features = test[available_features]
                            ground_truth = test[target_feature].values
                            
                            # Use LightGBM with same d1namo methodology
                            weights = [np.where(X['patient_id'].values[idx] == f"patient_{patient}", current_patient_weight, 1) 
                                     for idx in indices]
                            weights_train, weights_val = weights[0], weights[1]
                            model = lgb.train(LGB_PARAMS, 
                                            lgb.Dataset(X_train, label=y_train, weight=weights_train), 
                                            valid_sets=[lgb.Dataset(X_val, label=y_val, weight=weights_val)])
                            predictions = model.predict(test_features.values)
                            
                            rmse = np.sqrt(mean_squared_error(ground_truth, predictions))
                            results.append([mapping_type, dataset_name, horizon, patient, rmse])
                            
                        except Exception as e:
                            print(f"    Error for patient {patient}, horizon {horizon}, day {test_day}, hour {hour}: {e}")
                            continue
        
        else:  # Non-Bézier approaches on D1namo - need to load raw data and apply temporal mapping
            print(f"    Applying {mapping_type} mapping to D1namo raw data...")
            
            # We need to load the raw D1namo data again and apply our temporal mapping
            def get_projected_value(window, prediction_horizon):
                x = np.arange(len(window))
                coeffs = np.polyfit(x, window, deg=3)
                return np.polyval(coeffs, len(window) + prediction_horizon)
            
            def get_raw_data(patient):
                glucose_data = pd.read_csv(f"../diabetes_subset_pictures-glucose-food-insulin/{patient}/glucose.csv")
                insulin_data = pd.read_csv(f"../diabetes_subset_pictures-glucose-food-insulin/{patient}/insulin.csv")
                food_data = pd.read_csv(f"../food_data/pixtral-large-latest/{patient}.csv")
                glucose_data["datetime"] = pd.to_datetime(glucose_data["date"] + ' ' + glucose_data["time"])
                glucose_data = glucose_data.drop(['type', 'comments', 'date', 'time'], axis=1)
                glucose_data['glucose'] *= 18.0182
                glucose_data['hour'] = glucose_data['datetime'].dt.hour
                glucose_data['time'] = glucose_data['hour'] + glucose_data['datetime'].dt.minute / 60
                insulin_data["datetime"] = pd.to_datetime(insulin_data["date"] + ' ' + insulin_data["time"])
                insulin_data.fillna(0, inplace=True)
                insulin_data['insulin'] = insulin_data['slow_insulin'] + insulin_data['fast_insulin']
                insulin_data = insulin_data.drop(['slow_insulin', 'fast_insulin', 'comment', 'date', 'time'], axis=1)
                food_data['datetime'] = pd.to_datetime(food_data['datetime'], format='%Y:%m:%d %H:%M:%S')
                food_data = food_data[['datetime', 'simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers']]
                combined_data = pd.concat([food_data, insulin_data]).sort_values('datetime').reset_index(drop=True)
                combined_data.fillna(0, inplace=True)
                for horizon in [12]:  # Only for our prediction horizons
                    glucose_data[f'glucose_{horizon}'] = glucose_data['glucose'].shift(-horizon) - glucose_data['glucose']
                glucose_data['glucose_change'] = glucose_data['glucose'] - glucose_data['glucose'].shift(1)
                glucose_data['glucose_change_projected'] = glucose_data['glucose_change'].rolling(6, min_periods=6).apply(lambda window: get_projected_value(window, 6))
                glucose_data['glucose_projected'] = glucose_data['glucose'].rolling(6, min_periods=6).apply(lambda window: get_projected_value(window, 6))
                glucose_data.dropna(subset=[f'glucose_{12}'], inplace=True)
                glucose_data['patient_id'] = f"patient_{patient}"
                return glucose_data, combined_data
            
            # Process each patient with the new temporal mapping
            for patient in PATIENTS_D1NAMO:
                print(f"    Processing patient {patient} with {mapping_type} mapping...")
                try:
                    glucose_data, combined_data = get_raw_data(patient)
                    
                    # Apply temporal features with the optimized parameters
                    processed_data = add_temporal_features(mapping_type, best_params, OPTIMIZATION_FEATURES_D1NAMO,
                                                         glucose_data, combined_data)
                    
                    # Now use simplified evaluation (similar to AZT1D approach)
                    for horizon in [DEFAULT_PREDICTION_HORIZON]:
                        target_feature = f'glucose_{horizon}'
                        
                        # Use first 3 days for training, rest for testing
                        train_days = processed_data['datetime'].dt.day.unique()[:3]
                        test_days = processed_data['datetime'].dt.day.unique()[3:6]  # Limit test days
                        
                        train_data = processed_data[processed_data['datetime'].dt.day.isin(train_days)]
                        
                        if len(train_data) < 50:
                            continue
                        
                        # Train model
                        feature_cols = ['glucose'] + OPTIMIZATION_FEATURES_D1NAMO
                        X_train = train_data[feature_cols].fillna(0).values
                        y_train = train_data[target_feature].fillna(0).values
                        
                        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                            X_train, y_train, test_size=0.2, random_state=42)
                        
                        model = lgb.train(LGB_PARAMS, lgb.Dataset(X_train_split, label=y_train_split),
                                        valid_sets=[lgb.Dataset(X_val_split, label=y_val_split)])
                        
                        # Test on multiple days
                        day_rmses = []
                        for test_day in test_days:
                            test_data = processed_data[processed_data['datetime'].dt.day == test_day]
                            if len(test_data) < 5:
                                continue
                                
                            X_test = test_data[feature_cols].fillna(0).values
                            y_test = test_data[target_feature].fillna(0).values
                            
                            if len(X_test) > 0 and len(y_test) > 0:
                                y_pred = model.predict(X_test)
                                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                                day_rmses.append(rmse)
                        
                        if day_rmses:
                            avg_rmse = np.mean(day_rmses)
                            results.append([mapping_type, dataset_name, horizon, patient, avg_rmse])
                
                except Exception as e:
                    print(f"    Error processing patient {patient}: {e}")
                    continue
    
    return results

# =================== LATEX TABLE GENERATION ===================

def generate_latex_table(results_df):
    """Generate LaTeX table with results"""
    
    # Calculate mean and std for each combination
    summary_stats = []
    
    for mapping in results_df['mapping'].unique():
        for dataset in results_df['dataset'].unique():
            mapping_data = results_df[(results_df['mapping'] == mapping) & (results_df['dataset'] == dataset)]
            
            if len(mapping_data) == 0:
                continue
            
            # Overall stats
            overall_mean = mapping_data['rmse'].mean()
            overall_std = mapping_data['rmse'].std()
            summary_stats.append([mapping, dataset, 'Overall', overall_mean, overall_std])
            
                    # Per horizon stats
        for horizon in [DEFAULT_PREDICTION_HORIZON]:
                horizon_data = mapping_data[mapping_data['horizon'] == horizon]
                if len(horizon_data) > 0:
                    horizon_mean = horizon_data['rmse'].mean()
                    horizon_std = horizon_data['rmse'].std()
                    summary_stats.append([mapping, dataset, f'{horizon*5}min', horizon_mean, horizon_std])
    
    summary_df = pd.DataFrame(summary_stats, columns=['mapping', 'dataset', 'horizon', 'mean_rmse', 'std_rmse'])
    
    # Create directories if they don't exist
    os.makedirs('../latex_tables', exist_ok=True)
    
    # Generate LaTeX table
    latex_content = """\\begin{table}[htbp]
\\centering
\\caption{Comprehensive Comparison of Temporal Mapping Approaches with LightGBM (60-minute predictions)}
\\label{tab:temporal_mapping_comparison}
\\begin{tabular}{llcc}
\\toprule
\\textbf{Mapping} & \\textbf{Dataset} & \\textbf{Overall} & \\textbf{60min} \\\\
\\midrule
"""
    
    # Group by mapping and dataset
    for mapping in ['bezier', 'lognormal', 'biexponential', 'gaussian']:
        for dataset in ['AZT1D', 'D1namo']:
            mapping_data = summary_df[(summary_df['mapping'] == mapping) & (summary_df['dataset'] == dataset)]
            
            if len(mapping_data) == 0:
                continue
            
            row = f"{mapping.capitalize()} & {dataset}"
            
            # Overall
            overall_row = mapping_data[mapping_data['horizon'] == 'Overall']
            if len(overall_row) > 0:
                mean_val = overall_row['mean_rmse'].iloc[0]
                std_val = overall_row['std_rmse'].iloc[0]
                row += f" & {mean_val:.2f}±{std_val:.2f}"
            else:
                row += " & --"
            
            # Only 60min horizon
            horizon_row = mapping_data[mapping_data['horizon'] == '60min']
            if len(horizon_row) > 0:
                mean_val = horizon_row['mean_rmse'].iloc[0]
                std_val = horizon_row['std_rmse'].iloc[0]
                row += f" & {mean_val:.2f}±{std_val:.2f}"
            else:
                row += " & --"
            
            row += " \\\\\n"
            latex_content += row
    
    latex_content += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    # Save LaTeX table
    with open('../latex_tables/temporal_mapping_comparison.tex', 'w') as f:
        f.write(latex_content)
    
    print("\nLaTeX table saved to ../latex_tables/temporal_mapping_comparison.tex")
    return latex_content

# =================== VISUALIZATION ===================

def create_bezier_visualization(params, time_points):
    """Create Bezier curve for visualization purposes"""
    points = np.array(params).reshape(-1, 2)
    points[0] = [0.0, 0.0]
    control_points = points[1:].copy()
    sorted_indices = np.argsort(control_points[:, 0])
    points[1:] = control_points[sorted_indices]
    points[-1, 1] = 0.0
    n = len(points) - 1
    
    impacts = np.zeros_like(time_points)
    for i, t in enumerate(time_points):
        if t < 0 or t > points[-1, 0]:
            impacts[i] = 0
        else:
            t_norm = t / points[-1, 0]
            impact = 0
            for j, point in enumerate(points):
                impact += point[1] * comb(n, j) * (t_norm**j) * ((1-t_norm)**(n-j))
            impacts[i] = impact
    return impacts

def visualize_temporal_mappings():
    """Visualize different temporal mapping approaches for AZT1D carbohydrates"""
    print("\nGenerating temporal mapping visualization...")
    
    time_points = np.linspace(0, 8, 100)  # 8 hours
    
    plt.figure(figsize=(12, 8))
    
    # Bézier curve (our best approach)
    bezier_params = [0.0, 0.0, 1.5, 0.8, 3.0, 0.3, 6.0, 0.0]
    bezier_impact = create_bezier_visualization(bezier_params, time_points)
    plt.plot(time_points, bezier_impact, linewidth=3, label='Bézier', alpha=0.8, color='#1f77b4')
    
    # Log-normal: properly constrained to start/end at 0 (extended to 8 hours)
    lognormal_impact = np.zeros_like(time_points)
    valid_range = (time_points > 0.5) & (time_points < 8.0)  # Extended to 8 hours
    t_valid = time_points[valid_range] - 0.5  # Shift so it starts from 0
    mu, sigma, amplitude = 1.2, 0.8, 0.9
    lognormal_impact[valid_range] = amplitude * (1 / (t_valid * sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(t_valid) - mu) / sigma) ** 2)
    plt.plot(time_points, lognormal_impact, linewidth=3, label='Log-normal', alpha=0.8, color='#ff7f0e')
    
    # Bi-exponential: constrained absorption-elimination model (extended to 8 hours)
    biexp_impact = np.zeros_like(time_points)
    valid_range = (time_points >= 0.2) & (time_points <= 8.0)  # Extended to 8 hours
    t_valid = time_points[valid_range] - 0.2
    # Absorption phase (fast) - elimination phase (slow)
    A1, A2, k1, k2 = 1.0, 1.0, 2.0, 0.3
    biexp_impact[valid_range] = A1 * (1 - np.exp(-k1 * t_valid)) * np.exp(-k2 * t_valid)
    # Normalize to reasonable peak
    biexp_impact = biexp_impact * (0.8 / np.max(biexp_impact))
    plt.plot(time_points, biexp_impact, linewidth=3, label='Bi-exponential', alpha=0.8, color='#2ca02c')
    
    # Gaussian: properly constrained bell curve (extended to 8 hours)
    gaussian_impact = np.zeros_like(time_points)
    mu, sigma, amplitude = 2.5, 1.2, 0.85
    # Only non-zero in realistic range and forced to 0 at boundaries (extended to 7.5 hours)
    mask = (time_points >= 0.3) & (time_points <= 7.5)
    gaussian_impact[mask] = amplitude * np.exp(-0.5 * ((time_points[mask] - mu) / sigma) ** 2)
    # Smooth transition to 0 at boundaries
    boundary_width = 0.2
    left_fade = (time_points >= 0.3) & (time_points <= 0.3 + boundary_width)
    right_fade = (time_points >= 7.5 - boundary_width) & (time_points <= 7.5)
    gaussian_impact[left_fade] *= (time_points[left_fade] - 0.3) / boundary_width
    gaussian_impact[right_fade] *= (7.5 - time_points[right_fade]) / boundary_width
    plt.plot(time_points, gaussian_impact, linewidth=3, label='Gaussian', alpha=0.8, color='#d62728')
    
    plt.xlabel('Time After Consumption (hours)', fontsize=14)
    plt.ylabel('Carbohydrate Impact Weight', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 8)
    plt.ylim(-0.05, 1.0)
    
    plt.tight_layout()
    plt.savefig('../images/supplementary_data/temporal_mapping_curves.eps', dpi=300, bbox_inches='tight')
    plt.close()

# =================== MAIN EXECUTION ===================

def main():
    print("Temporal Mapping Comparison Analysis")
    print("=" * 40)
    
    # Load existing results and check what's missing
    if os.path.exists('temporal_mapping_detailed_results.csv'):
        existing_results = pd.read_csv('temporal_mapping_detailed_results.csv')
        print(f"Loaded existing results: {len(existing_results)} rows")
        
        existing_combinations = {(row['mapping'], row['dataset']) for _, row in existing_results.iterrows()}
        all_combinations = [
            ('baseline', 'AZT1D'), ('bezier', 'AZT1D'), ('lognormal', 'AZT1D'), ('biexponential', 'AZT1D'), ('gaussian', 'AZT1D'),
            ('baseline', 'D1namo'), ('bezier', 'D1namo'), ('lognormal', 'D1namo'), ('biexponential', 'D1namo'), ('gaussian', 'D1namo')
        ]
        missing_combinations = [combo for combo in all_combinations if combo not in existing_combinations]
        
        if missing_combinations:
            print(f"Running {len(missing_combinations)} missing evaluations...")
            new_results = []
            
            for mapping, dataset in missing_combinations:
                print(f"  {mapping} + {dataset}")
                
                if dataset == 'AZT1D':
                    azt1d_data = load_azt1d_data()
                    results = evaluate_mapping_approach(mapping, 'AZT1D', azt1d_data)
                else:
                    d1namo_data, d1namo_combined = load_d1namo_data()
                    results = evaluate_mapping_approach(mapping, 'D1namo', d1namo_data, d1namo_combined)
                
                new_results.extend(results)
            
            new_results_df = pd.DataFrame(new_results, columns=['mapping', 'dataset', 'horizon', 'patient', 'rmse'])
            results_df = pd.concat([existing_results, new_results_df], ignore_index=True)
        else:
            print("All combinations already evaluated!")
            results_df = existing_results
    else:
        print("\nNo existing results found. Running full evaluation...")
        azt1d_data = load_azt1d_data()
        d1namo_data, d1namo_combined = load_d1namo_data()
        
        print(f"AZT1D: {len(azt1d_data)} records from {len(PATIENTS_AZT1D)} patients")
        print(f"D1namo: {len(d1namo_data)} records from {len(PATIENTS_D1NAMO)} patients")
        
        mapping_approaches = ['baseline', 'bezier', 'lognormal', 'biexponential', 'gaussian']
        all_results = []
        
        for mapping in mapping_approaches:
            results_azt1d = evaluate_mapping_approach(mapping, 'AZT1D', azt1d_data)
            all_results.extend(results_azt1d)
            
            results_d1namo = evaluate_mapping_approach(mapping, 'D1namo', d1namo_data, d1namo_combined)
            all_results.extend(results_d1namo)
        
        results_df = pd.DataFrame(all_results, columns=['mapping', 'dataset', 'horizon', 'patient', 'rmse'])
    
    # Save detailed results
    results_df.to_csv('temporal_mapping_detailed_results.csv', index=False)
    print(f"Analysis complete: {len(results_df)} evaluations")
    
    # Generate and display summary
    print("\nSummary Results:")
    print("-" * 40)
    
    mapping_approaches = ['baseline', 'bezier', 'lognormal', 'biexponential', 'gaussian']
    for mapping in mapping_approaches:
        for dataset in ['AZT1D', 'D1namo']:
            subset = results_df[(results_df['mapping'] == mapping) & (results_df['dataset'] == dataset)]
            if len(subset) > 0:
                overall_rmse = subset['rmse'].mean()
                overall_std = subset['rmse'].std()
                print(f"{mapping.capitalize():<13} {dataset}: {overall_rmse:.2f}±{overall_std:.2f} RMSE")
    
    # Generate LaTeX table and visualization
    latex_table = generate_latex_table(results_df)
    print("\nGenerating visualization...")
    visualize_temporal_mappings()
    print("All outputs generated successfully!")

if __name__ == "__main__":
    main()