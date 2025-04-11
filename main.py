import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import os
import pickle
import optuna
from joblib import Parallel, delayed
from scipy.special import comb
import json
import warnings

# Silence warnings
warnings.filterwarnings('ignore')

# Constants
features = ['simple_sugars', 'complex_sugars', 'fats', 'dietary_fibers', 'proteins', 'insulin']
meal_features = ['simple_sugars', 'complex_sugars', 'fats', 'dietary_fibers', 'proteins']
features_to_remove = ['glucose_next', 'datetime', 'hour']
patients = ['001', '002', '004', '006', '007', '008']
approaches = ['pixtral-large-latest', 'nollm']
prediction_horizons = [6, 9, 12, 18, 24]

# Optimization parameters
train_size = 0.9
n_trials = 300
random_seed = 42
n_jobs = 6
load_params=False

# LightGBM parameters
lgb_params = {
    'max_depth': 3,
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'min_data_in_leaf': 0,
    'objective': 'regression',
    'data_sample_strategy': 'goss',
    'use_quantized_grad': True,
    'random_state': 42,
    'deterministic': True,
    'force_row_wise': True,
    'num_threads': 20,
    'reg_alpha': 0,
    'path_smooth': 1,
    'reg_lambda': 20,
    'verbosity': -1,
}
model = lgb.LGBMRegressor(**lgb_params)
callbacks = [lgb.early_stopping(stopping_rounds=10, verbose=False)]

def bezier_curve(points, num=50):
    """Generate Bezier curve from control points using Bernstein polynomials"""
    n = len(points) - 1  # Degree of curve is n
    t = np.linspace(0, 1, num)
    curve = np.zeros((num, 2))
    
    for i, point in enumerate(points):
        # Calculate Bernstein polynomial basis
        curve += np.outer(comb(n, i) * (t**i) * ((1-t)**(n-i)), point)
    
    return curve[np.argsort(curve[:, 0])]

def get_projected_value(window, prediction_horizon):
    x = np.arange(len(window))
    coeffs = np.polyfit(x, window, deg=3)
    return np.polyval(coeffs, len(window) + prediction_horizon)

def get_data(patient, prediction_horizon):
    # Load data
    glucose_data = pd.read_csv(f"diabetes_subset_pictures-glucose-food-insulin/{patient}/glucose.csv")
    insulin_data = pd.read_csv(f"diabetes_subset_pictures-glucose-food-insulin/{patient}/insulin.csv")
    food_data = pd.read_csv(f"food_data/pixtral-large-latest/{patient}.csv")

    # Process glucose data
    glucose_data["datetime"] = pd.to_datetime(glucose_data["date"] + ' ' + glucose_data["time"])
    glucose_data = glucose_data.drop(['type', 'comments', 'date', 'time'], axis=1)
    glucose_data['glucose'] *= 18.0182
    glucose_data['hour'] = glucose_data['datetime'].dt.hour
    glucose_data['time'] = glucose_data['hour'] + glucose_data['datetime'].dt.minute / 60

    # Process insulin data
    insulin_data["datetime"] = pd.to_datetime(insulin_data["date"] + ' ' + insulin_data["time"])
    insulin_data['insulin'] = insulin_data['slow_insulin'] + insulin_data['fast_insulin']
    insulin_data = insulin_data.drop(['slow_insulin', 'fast_insulin', 'comment', 'date', 'time'], axis=1)

    # Process food data
    food_data['datetime'] = pd.to_datetime(food_data['datetime'], format='%Y:%m:%d %H:%M:%S')
    food_data = food_data[['datetime', 'simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers']]

    # Combine data
    combined_data = pd.concat([food_data, insulin_data]).sort_values('datetime').reset_index(drop=True)
    combined_data.fillna(0, inplace=True)

    # Calculate target variables
    glucose_data['glucose_next'] = glucose_data['glucose'] - glucose_data['glucose'].shift(-prediction_horizon)
    glucose_data['glucose_change'] = glucose_data['glucose'] - glucose_data['glucose'].shift(1)
    
    window_size = 6
    glucose_data['glucose_change_projected'] = glucose_data['glucose_change'].rolling(
        window=window_size, min_periods=window_size
    ).apply(lambda window: get_projected_value(window, prediction_horizon))
    
    glucose_data['glucose_projected'] = glucose_data['glucose'].rolling(
        window=window_size, min_periods=window_size
    ).apply(lambda window: get_projected_value(window, prediction_horizon))
    
    glucose_data.dropna(subset=['glucose_next'], inplace=True)
    return glucose_data, combined_data

def add_features(params, features, data, prediction_horizon):
    glucose_data, combined_data = data
    
    # Convert datetime to nanoseconds for efficient vectorized operations
    glucose_times = glucose_data['datetime'].values.astype('datetime64[s]').astype(np.int64)
    combined_times = combined_data['datetime'].values.astype('datetime64[s]').astype(np.int64)
    
    # Calculate time difference matrix (in hours)
    time_diff_hours = ((glucose_times[:, None] - combined_times[None, :]) / 3600)
    
    for feature in features:
        
        # Generate Bezier curve
        curve = bezier_curve(np.array(params[feature]).reshape(-1, 2), num=100)
        x_curve, y_curve = curve[:, 0], curve[:, 1]
        
        # Create weights array
        weights = np.zeros_like(time_diff_hours)
        
        # For each time difference, find the closest point on bezier curve
        for i in range(len(glucose_times)):
            for j in range(len(combined_times)):
                if time_diff_hours[i, j] >= 0 and time_diff_hours[i, j] <= max(x_curve):
                    # Find closest x value in curve
                    idx = np.abs(x_curve - time_diff_hours[i, j]).argmin()
                    weights[i, j] = y_curve[idx]
        
        # Compute impact and shift by prediction horizon
        feature_values = pd.Series(np.dot(weights, combined_data[feature].values))
        glucose_data[feature] = feature_values.shift(-prediction_horizon) - feature_values
    return glucose_data

def optimize_for_patient(patient, prediction_horizon, base_control_points):
    """Optimize parameters for a single patient"""
    glucose_data, combined_data = get_data(patient, prediction_horizon)
    first_days = glucose_data['datetime'].dt.day.unique()[:3]
    mask = glucose_data['datetime'].dt.day.isin(first_days)
    train_glucose_data = glucose_data[mask].copy()
    data = (train_glucose_data, combined_data) # Pass only training glucose data for optimization
    
    # --- Start Pre-calculation ---
    # Pre-calculate normalized target as it's constant across trials for this patient
    target = train_glucose_data['glucose_next'].copy()
    target_mean = target.mean()
    target_std = target.std()
    normalized_target = (target - target_mean) / target_std
    # --- End Pre-calculation ---
    
    # Create Optuna optimization study
    study = optuna.create_study(study_name=patient, direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_seed), storage=f"sqlite:///optuna_{patient}.db", load_if_exists=True)
    
    # Define the objective function for Optuna
    def objective(trial):
        # Container for parameters and weights
        params = {}
        weights = {}
        
        # Generate parameters and weights for each feature
        for feature in features:
            base_bounds = base_control_points[feature]
            feature_params = []
            
            # First point is fixed at origin (0,0)
            feature_params.extend([0.0, 0.0])
            
            # Second control point (x and y optimized)
            x2 = trial.suggest_float(f"{feature}_x1", base_bounds[0][0], base_bounds[0][1])
            y2 = trial.suggest_float(f"{feature}_y1", 0.1, 1.0)
            feature_params.extend([x2, y2])
            
            # Third control point (x and y optimized)
            min_x3 = x2 + 0.1
            x3 = trial.suggest_float(f"{feature}_x2", min_x3, base_bounds[1][1])
            y3 = trial.suggest_float(f"{feature}_y2", 0.0, 1.0) # y can be lower/zero
            feature_params.extend([x3, y3])

            # Fourth control point (x optimized, y=0)
            min_x4 = x3 + 0.1
            x4 = trial.suggest_float(f"{feature}_x3", min_x4, base_bounds[2][1])
            feature_params.extend([x4, 0.0])
            
            params[feature] = feature_params
            # --- Start Weight Suggestion ---
            weights[feature] = trial.suggest_float(f"{feature}_weight", -1.0, 1.0)
            # --- End Weight Suggestion ---
        
        # Process data with current parameter set
        # Important: Use the original 'data' tuple which contains the train_glucose_data
        glucose_data_copy, combined_data_copy = data[0].copy(), data[1].copy() 
        df = add_features(params, features, (glucose_data_copy, combined_data_copy), prediction_horizon)

        # --- Use Pre-calculated Normalized Target --- 
        feature_impacts = df[features].copy()

        # Normalize features and apply weights (this depends on trial params, so must be inside)
        weighted_normalized_sum = pd.Series(np.zeros(len(df)), index=df.index)
        for feature in features:
            col = feature_impacts[feature]
            col_mean = col.mean()
            col_std = col.std()
            if col_std == 0: 
                 normalized_col = pd.Series(np.zeros(len(df)), index=df.index)
            else:
                 normalized_col = (col - col_mean) / col_std
            
            weighted_normalized_sum += normalized_col * weights[feature]

        # Calculate the absolute Pearson correlation using pre-calculated normalized_target
        # Ensure indices align if add_features modified the index (it shouldn't currently)
        correlation = normalized_target.corr(weighted_normalized_sum)

        if pd.isna(correlation):
            # Return penalty if correlation is NaN (e.g., target or weighted sum is constant)
            return 1.0 

        return 1.0 - abs(correlation)
        # --- End Change ---
    
    # Run optimization with parallelize flag
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    
    # Get best parameters and weights
    best_params = {'bezier_points': {}, 'weights': {}}
    for feature in features:
        feature_params = [0.0, 0.0]  # First point fixed at origin
        
        # Bezier control points
        feature_params.append(study.best_params[f"{feature}_x1"])
        feature_params.append(study.best_params[f"{feature}_y1"])
        feature_params.append(study.best_params[f"{feature}_x2"])
        feature_params.append(study.best_params[f"{feature}_y2"])
        feature_params.append(study.best_params[f"{feature}_x3"])
        feature_params.append(0.0)  # y4 is fixed at 0
        best_params['bezier_points'][feature] = feature_params
        
        # Weights
        best_params['weights'][feature] = study.best_params[f"{feature}_weight"]
    
    print(f"Completed optimization for patient {patient}, best score (1 - abs(corr)): {study.best_value:.4f}")
    # Optionally print weights
    # print(f"Best weights for patient {patient}: {best_params['weights']}")
    os.remove(f"optuna_{patient}.db")
        
    return patient, best_params

# Format: [[min_x1, max_x1], [min_x2, max_x2], [min_x3, max_x3]] - Bounds for x coordinates of the 3 optimized points P1, P2, P3.
# y1 bounds: [0.1, 1.0]
# y2 bounds: [0.0, 1.0]
# P0 is (0,0), P4 is (x3, 0)
base_control_points = {
    'simple_sugars': [[0.1, 0.8], [0.5, 1.5], [2.0, 4.0]],   # Fast rise, peak ~1h, return 2-4h
    'complex_sugars': [[0.5, 1.5], [1.5, 3.0], [4.0, 7.0]],  # Slower rise, peak 1.5-3h, return 4-7h
    'proteins': [[1.0, 2.5], [2.5, 5.0], [5.0, 10.0]],       # Slow rise, peak 2.5-5h, return 5-10h
    'fats': [[1.5, 3.0], [3.0, 6.0], [6.0, 14.0]],          # Slowest rise, peak 3-6h, return 6-14h
    'dietary_fibers': [[1.0, 3.0], [3.0, 6.0], [6.0, 18.0]], # Blunting effect, long duration
    'insulin': [[0.1, 0.5], [0.5, 1.5], [1.5, 4.0]],         # Fast action, peak ~1h, return 1.5-4h
}

# Main evaluation loop
df = pd.DataFrame(columns=['Approach', 'Prediction Horizon', 'Patient', 'Day', 'Hour', 'RMSE']) 

# Call visualization after loading or optimizing parameters
if load_params:
    with open('parameters/patient_bezier_params.json', 'r') as f:
        patient_params = json.load(f)
else:
    results = Parallel(n_jobs=min(len(patients), n_jobs))(delayed(optimize_for_patient)(patient, 6, base_control_points) for patient in patients)
    patient_params = dict(results)
    # Save as JSON (structure now includes bezier_points and weights)
    with open('parameters/patient_bezier_params.json', 'w') as f:
        # No need for special conversion for weights, they are floats
        json.dump(patient_params, f, indent=2)

for approach in approaches:
    for prediction_horizon in prediction_horizons:
        
        # Process per patient for evaluation
        for patient in patients:
            # Get patient-specific feature parameters (access bezier points specifically)
            bezier_params = patient_params[patient]['bezier_points']
            # Note: Weights are not used here in the current evaluation logic
            data = add_features({k: v for k, v in bezier_params.items() if k in features}, features, get_data(patient, prediction_horizon), prediction_horizon)
            
            if approach == 'nollm':
                data.drop(meal_features, axis=1, inplace=True)
            else:
                data.to_csv(f'data/{prediction_horizon}_{patient}.csv', index=False)
            
            all_preds = []
            all_test_data = []
            days = data['datetime'].dt.day.unique()
            test_days = days[3:]  
            
            for test_day in test_days:
                day_mask = data['datetime'].dt.day == test_day
                test_day_data = data[day_mask]
                hours = test_day_data['hour'].unique()
                
                for hour in hours:
                    hour_mask = test_day_data['hour'] == hour
                    test = test_day_data[hour_mask]
                    
                    # Train on all data before the current hour from any day
                    earliest_test_time = test['datetime'].min()
                    safe_train_mask = data['datetime'].shift(-prediction_horizon) < earliest_test_time
                    train = data[safe_train_mask]
                    X_train, X_val, y_train, y_val = train_test_split(train.drop(features_to_remove, axis=1), train['glucose_next'], train_size=train_size, random_state=42)
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks, eval_metric='rmse')
                    hour_preds = model.predict(test.drop(features_to_remove, axis=1))
                    rmse = np.sqrt(mean_squared_error(test['glucose_next'], hour_preds))
                    
                    all_preds.append(hour_preds)
                    all_test_data.append(test)
                    
                    df = pd.concat([df, pd.DataFrame({
                        'Approach': [approach],
                        'Prediction Horizon': [prediction_horizon],
                        'Patient': [patient],
                        'Day': [test_day],
                        'Hour': [hour],
                        'RMSE': [rmse]
                    })], ignore_index=True)
                    
                    if test_day == test_days[-1] and hour == hours[-1]:
                        
                        # Create directory if it doesn't exist
                        os.makedirs(f'models/{approach}/{prediction_horizon}', exist_ok=True)
                        model_filename = f'models/{approach}/{prediction_horizon}/patient_{patient}_model.pkl'
                        with open(model_filename, 'wb') as file:
                            pickle.dump(model, file)
            
            combined_preds = np.concatenate(all_preds)
            combined_test = pd.concat(all_test_data)
            predictions = pd.DataFrame({
                'Predictions': combined_test['glucose'] - combined_preds, 
                'Ground_truth': combined_test['glucose'] - combined_test['glucose_next'], 
                'Datetime': combined_test['datetime']
            })
            # Create directory if it doesn't exist
            os.makedirs(f'predictions/{approach}/{prediction_horizon}', exist_ok=True)
            predictions.to_csv(f'predictions/ {approach}/{prediction_horizon}/{patient}_predictions.csv', index=False)
        print(f"Average RMSE for {approach}, prediction horizon {prediction_horizon}: {df[(df['Approach'] == approach) & (df['Prediction Horizon'] == prediction_horizon)]['RMSE'].mean():.4f}")

df.to_csv('results/evaluation_metrics.csv', index=False) 