import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import os
import pickle
import logging
import pyswarms as ps
from scipy.special import comb
import json
import warnings

# Silence warnings
logging.getLogger('optuna').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

# Constants
features = ['simple_sugars', 'complex_sugars', 'fats', 'dietary_fibers', 'proteins', 'fast_insulin', 'slow_insulin']
meal_features = ['simple_sugars', 'complex_sugars', 'fats', 'dietary_fibers', 'proteins']
insulin_features = ['fast_insulin', 'slow_insulin']
carb_features = ['simple_sugars', 'complex_sugars']
fat_protein_features = ['fats', 'proteins']
features_to_remove = ['glucose_next', 'datetime']
patients = ['001', '002', '004', '006', '007', '008']
approaches = ['pixtral-large-latest', 'nollm']
prediction_horizons = [6, 9, 12, 18, 24]

# Optimization parameters
train_size = 0.9
n_particles = 100
iters = 10

# LightGBM parameters
lgb_params = {
    'max_depth': 3,
    'n_estimators': 1000,
    'learning_rate': 0.1,
    'objective': 'regression',
    'data_sample_strategy': 'goss',
    'random_state': 42,
    'deterministic': True,
    'use_quantized_grad': True,
    'force_row_wise': True,
    'num_threads': 5,
    'verbosity': -1,
    'reg_alpha': 10,
    'min_child_samples': 10,
}
model = lgb.LGBMRegressor(**lgb_params)
callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]

def bernstein_poly(i, n, t):
    """Bernstein polynomial basis for Bezier curves"""
    return comb(n, i) * (t**i) * ((1-t)**(n-i))

def bezier_curve(points, num=50):
    """Generate Bezier curve from control points"""
    n = len(points) - 1  # Degree of curve is n
    t = np.linspace(0, 1, num)
    curve = np.zeros((num, 2))
    
    for i, point in enumerate(points):
        curve += np.outer(bernstein_poly(i, n, t), point)
    
    return curve

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

    # Process insulin data
    insulin_data["datetime"] = pd.to_datetime(insulin_data["date"] + ' ' + insulin_data["time"])
    insulin_data = insulin_data.drop(['comment', 'date', 'time'], axis=1)

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

def add_features_bezier(params, features, preprocessed_data, prediction_horizon, patient):
    glucose_data, combined_data = preprocessed_data[patient]
    
    # Convert datetime to nanoseconds for efficient vectorized operations
    glucose_times = glucose_data['datetime'].values.astype('datetime64[s]').astype(np.int64)
    combined_times = combined_data['datetime'].values.astype('datetime64[s]').astype(np.int64)
    
    # Calculate time difference matrix (in hours)
    time_diff_hours = ((glucose_times[:, None] - combined_times[None, :]) / 3600)
    
    for feature in features:
        # Extract control points from params
        control_points = np.array(params[feature]).reshape(-1, 2)
        
        # Generate Bezier curve
        curve = bezier_curve(control_points, num=100)
        x_curve, y_curve = curve[:, 0], curve[:, 1]
        
        # Create weights array
        weights = np.zeros_like(time_diff_hours, dtype=np.float32)
        
        # For each time difference, find the closest point on bezier curve
        for i in range(len(glucose_times)):
            for j in range(len(combined_times)):
                if time_diff_hours[i, j] >= 0 and time_diff_hours[i, j] <= max(x_curve):
                    # Find closest x value in curve
                    idx = np.abs(x_curve - time_diff_hours[i, j]).argmin()
                    weights[i, j] = y_curve[idx]
        
        # Compute impact and shift by prediction horizon
        glucose_data[feature] = np.dot(weights, combined_data[feature].values)
        glucose_data[feature] = glucose_data[feature] -glucose_data[feature].shift(-prediction_horizon)
    
    return glucose_data

def objective_bezier(params, data, features, patient, prediction_horizon, train_size, callbacks):
    results = []
    
    for p in params:
        # Evaluate valid parameters
        result = _evaluate_bezier_params(p, data, features, patient, prediction_horizon, train_size, callbacks)
        results.append(result)
    
    return np.array(results)

def _evaluate_bezier_params(params, data, features, patient, prediction_horizon, train_size, callbacks):
    """Helper function to evaluate a single parameter set using Bezier curve mapping"""
    # Reshape params into feature-specific control points
    feature_params = {}
    points_per_feature = 8  # 4 control points with x,y coordinates
    
    for i, feature in enumerate(features):
        idx = i * points_per_feature
        if idx + points_per_feature <= len(params):
            feature_params[feature] = params[idx:idx + points_per_feature]
    
    # Process data with current parameter set
    glucose_data, combined_data = data
    processed_data = add_features_bezier(feature_params, features, {patient: (glucose_data, combined_data)}, 
                                prediction_horizon, patient)
    
    # Check for and handle NaN values
    processed_data = processed_data.fillna(0)
    
    # Split data into train/validation sets
    X = processed_data.drop(features_to_remove, axis=1)
    y = processed_data['glucose_next']
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=train_size, random_state=42
    )
    
    # Train model
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks
    )
    
    # Make predictions and calculate RMSE
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    return rmse

def optimize_patient_parameters_bezier(patients, prediction_horizon, base_control_points):
    patient_feature_params = {}
    
    for patient in patients:
        # Extract first 3 days for training/optimization
        data = get_data(patient, prediction_horizon)
        glucose_data, combined_data = data
        first_days = glucose_data['datetime'].dt.day.unique()[:3]
        mask = glucose_data['datetime'].dt.day.isin(first_days)
        data = (glucose_data[mask].copy(), combined_data)
        
        # Define bounds for control points optimization
        lb = []  # Lower bounds
        ub = []  # Upper bounds
        
        for feature in features:
            base_points = base_control_points[feature]
            
            # For each control point (x,y), set bounds
            for i in range(0, len(base_points), 2):
                x_val, y_val = base_points[i], base_points[i+1]
                
                # X coordinates (time)
                if i == 0:  # First point is always at origin
                    lb.append(0.0)
                    ub.append(0.0)
                else:
                    lb.append(max(0.0, 0.5 * x_val))
                    ub.append(min(24.0, 1.5 * x_val))
                
                # Y coordinates (effect magnitude)
                if i == 0:  # First point is always at zero effect
                    lb.append(0.0)
                    ub.append(0.0)
                else:
                    lb.append(max(0.0, 0.5 * y_val))
                    ub.append(min(1.0, 1.5 * y_val))
                    
        bounds = (np.array(lb), np.array(ub))
        
        # Initialize optimizer with Global Best PSO (no KDTree)
        optimizer = ps.single.GlobalBestPSO(
            n_particles=n_particles, 
            dimensions=len(features) * 8,
            options={
                'c1': 0.7,
                'c2': 0.5,
                'w': 0.8
            }, 
            bounds=bounds
        )
        
        # Objective function
        def obj_func(params):
            return objective_bezier(params, data, features, patient, prediction_horizon, train_size, callbacks)
        
        # Optimize
        cost, pos = optimizer.optimize(obj_func, iters=iters, verbose=False)
        
        # Extract best parameters
        best_params = {}
        for i, feature in enumerate(features):
            idx = i * 8
            best_params[feature] = pos[idx:idx+8]
        
        patient_feature_params[patient] = best_params
        print(f"Completed optimization for patient {patient}, best RMSE: {cost:.4f}")
            
    return patient_feature_params

# Initial control points for Bezier curves
# Format: [x1, y1, x2, y2, x3, y3, x4, y4] - Four control points per feature
base_control_points = {
    'simple_sugars': [0.0, 0.0, 0.5, 0.8, 1.0, 0.5, 2.0, 0.0],       # Fast rise, quick drop
    'complex_sugars': [0.0, 0.0, 1.0, 0.5, 2.0, 0.7, 4.0, 0.0],      # Slower rise, longer effect
    'proteins': [0.0, 0.0, 2.0, 0.3, 4.0, 0.6, 8.0, 0.0],           # Very slow rise, extended effect
    'fats': [0.0, 0.0, 3.0, 0.2, 6.0, 0.4, 10.0, 0.0],              # Slowest rise, longest effect
    'dietary_fibers': [0.0, 0.0, 0.5, 0.1, 2.0, 0.3, 5.0, 0.0],      # Moderate effect curve
    'fast_insulin': [0.0, 0.0, 0.3, 0.9, 1.0, 0.3, 2.0, 0.0],        # Fast action, quick drop
    'slow_insulin': [0.0, 0.0, 1.0, 0.4, 3.0, 0.7, 6.0, 0.0]         # Slower action, extended effect
}

# Run optimization for prediction horizon 6
patient_params = optimize_patient_parameters_bezier(patients, 6, base_control_points)
# Save the optimized parameters
os.makedirs('parameters', exist_ok=True)
with open('parameters/patient_bezier_params.pkl', 'wb') as f:
    pickle.dump(patient_params, f)

# Optionally save as JSON for better readability
with open('parameters/patient_bezier_params.json', 'w') as f:
    # Convert numpy arrays to lists for JSON serialization
    json_params = {}
    for patient, features in patient_params.items():
        json_params[patient] = {
            feature: params.tolist() if isinstance(params, np.ndarray) else params
            for feature, params in features.items()
        }
    json.dump(json_params, f, indent=2)

# Main evaluation loop
df = pd.DataFrame(columns=['Approach', 'Prediction Horizon', 'Patient', 'Day', 'Hour', 'RMSE']) 

for approach in approaches:
    for prediction_horizon in prediction_horizons:
        
        # Get raw data for all patients
        data = {patient: get_data(patient, prediction_horizon) for patient in patients}
        
        # Create directories for models
        os.makedirs(f'models/{approach}/{prediction_horizon}', exist_ok=True)
        
        # Process per patient for evaluation
        for patient in patients:
            # Get patient-specific feature parameters
            patient_feature_params = patient_params[patient].copy()
            processed_data = add_features_bezier(
                {k: v for k, v in patient_feature_params.items() if k in features},
                features,
                {patient: data[patient]}, 
                prediction_horizon, patient
            )
            
            if approach == 'nollm':
                processed_data.drop(meal_features, axis=1, inplace=True)
            
            all_preds = []
            all_test_data = []
            
            days = processed_data['datetime'].dt.day.unique()
            train_days = days[:3]
            test_days = days[3:]  
            
            for test_day in test_days:
                day_mask = processed_data['datetime'].dt.day == test_day
                test_day_data = processed_data[day_mask]
                hours = sorted(test_day_data['hour'].unique())
                
                for hour in hours:
                    hour_mask = test_day_data['hour'] == hour
                    test = test_day_data[hour_mask]
                    
                    prior_days_mask = processed_data['datetime'].dt.day < test_day
                    prior_hours_mask = (processed_data['datetime'].dt.day == test_day) & (processed_data['hour'] < hour)
                    
                    train = pd.concat([
                        processed_data[prior_days_mask],
                        processed_data[prior_hours_mask]
                    ])
                    
                    hour_model = lgb.LGBMRegressor(**lgb_params)
                    
                    X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(
                        train.drop(features_to_remove, axis=1), train['glucose_next'], 
                        np.ones(len(train)), train_size=train_size, random_state=42
                    )
                    
                    hour_model.fit(
                        X_train, y_train,
                        sample_weight=weights_train,
                        eval_set=[(X_val, y_val)], 
                        eval_sample_weight=[weights_val],
                        callbacks=callbacks
                    )
                    
                    hour_preds = hour_model.predict(test.drop(features_to_remove, axis=1))
                    hour_rmse = np.sqrt(mean_squared_error(test['glucose_next'], hour_preds))
                    
                    all_preds.append(hour_preds)
                    all_test_data.append(test)
                    
                    df = pd.concat([df, pd.DataFrame({
                        'Approach': [approach],
                        'Prediction Horizon': [prediction_horizon],
                        'Patient': [patient],
                        'Day': [test_day],
                        'Hour': [hour],
                        'RMSE': [hour_rmse]
                    })], ignore_index=True)
                    
                    if test_day == test_days[-1] and hour == hours[-1]:
                        model_filename = f'models/{approach}/{prediction_horizon}/patient_{patient}_model.pkl'
                        with open(model_filename, 'wb') as file:
                            pickle.dump(hour_model, file)
                
            combined_preds = np.concatenate(all_preds)
            combined_test = pd.concat(all_test_data)
            
            predictions = pd.DataFrame({
                'Predictions': combined_test['glucose'] - combined_preds, 
                'Ground_truth': combined_test['glucose'] - combined_test['glucose_next'], 
                'Datetime': combined_test['datetime']
            })
            
            os.makedirs(f'predictions/{approach}/{prediction_horizon}', exist_ok=True)
            predictions.to_csv(f'predictions/{approach}/{prediction_horizon}/{patient}_predictions.csv', index=False)
        
        current_metrics = df[(df['Approach'] == approach) & (df['Prediction Horizon'] == prediction_horizon)]
        current_rmse = current_metrics['RMSE'].mean()
        print(f"Average RMSE for {approach}, prediction horizon {prediction_horizon}: {current_rmse:.4f}")

# Save the final metrics dataframe
df.to_csv('evaluation_metrics_bezier.csv', index=False) 