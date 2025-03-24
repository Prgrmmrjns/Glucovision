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
n_trials = 100
random_seed = 42
n_jobs = 6

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
    
    # Sort curve by x-values to ensure function-like behavior
    indices = np.argsort(curve[:, 0])
    curve = curve[indices]
    
    # Remove any duplicate x-values (keeping the first occurrence)
    _, unique_indices = np.unique(curve[:, 0], return_index=True)
    unique_indices.sort()
    curve = curve[unique_indices]
    
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
        glucose_data[feature] = glucose_data[feature].shift(-prediction_horizon) - glucose_data[feature]
    
    return glucose_data

def optimize_for_patient(patient, prediction_horizon, base_control_points):
    """Optimize parameters for a single patient"""
    # Extract first 3 days for training/optimization
    data = get_data(patient, prediction_horizon)
    glucose_data, combined_data = data
    first_days = glucose_data['datetime'].dt.day.unique()[:3]
    mask = glucose_data['datetime'].dt.day.isin(first_days)
    data = (glucose_data[mask].copy(), combined_data)
    
    # Create Optuna optimization study with SQLite storage for parallelization
    storage_name = f"sqlite:///optuna_{patient}_ph{prediction_horizon}.db"
    study = optuna.create_study(
        study_name=f"patient_{patient}_ph{prediction_horizon}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_seed),
        storage=storage_name,
        load_if_exists=True
    )
    
    # Define the objective function for Optuna
    def objective(trial):
        # Container for parameters
        params = {}
        
        # Generate parameters for each feature
        for feature in features:
            base_points = base_control_points[feature]
            feature_params = []
            
            # First point is fixed at origin (0,0)
            feature_params.extend([0.0, 0.0])
            prev_x = 0.0
            
            # Generate remaining control points with constraints
            for i in range(2, len(base_points), 2):
                base_x, base_y = base_points[i], base_points[i+1]
                
                # X coordinates (time) - ensure monotonically increasing
                min_x = prev_x + 0.1
                x_val = trial.suggest_float(
                    f"{feature}_x{i//2}", 
                    min_x, 
                    min(24.0, 1.5 * base_x),
                )
                feature_params.append(x_val)
                prev_x = x_val
                
                # Y coordinates (effect magnitude)
                y_val = trial.suggest_float(
                    f"{feature}_y{i//2}", 
                    max(0.0, 0.5 * base_y), 
                    min(1.0, 1.5 * base_y),
                )
                feature_params.append(y_val)
            
            params[feature] = feature_params
        
        # Evaluate the parameters
        result = evaluate_bezier_params(params, data, features, patient, 
                                      prediction_horizon, train_size, callbacks)
        return result
    
    # Run optimization with parallelize flag
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    
    # Get best parameters
    best_params = {}
    for feature in features:
        feature_params = [0.0, 0.0]  # First point fixed at origin
        
        for i in range(2, 8, 2):
            x_key = f"{feature}_x{i//2}"
            y_key = f"{feature}_y{i//2}"
            feature_params.append(study.best_params[x_key])
            feature_params.append(study.best_params[y_key])
        
        best_params[feature] = feature_params
    
    print(f"Completed optimization for patient {patient}, best RMSE: {study.best_value:.4f}")
    
    # Clean up the database file
    try:
        os.remove(f"optuna_{patient}.db")
    except:
        pass
        
    return patient, best_params

def optimize_patient_parameters_bezier(patients, prediction_horizon, base_control_points):
    """Optimize parameters for all patients in parallel"""
    # Use joblib to parallelize across patients
    results = Parallel(n_jobs=min(len(patients), n_jobs))(
        delayed(optimize_for_patient)(patient, prediction_horizon, base_control_points)
        for patient in patients
    )
    
    # Convert results to dictionary
    patient_feature_params = dict(results)
    return patient_feature_params

def evaluate_bezier_params(params, data, features, patient, prediction_horizon, train_size, callbacks):
    """Thread-safe evaluation function"""
    # Create a local copy of the model to avoid sharing between threads
    local_model = lgb.LGBMRegressor(**lgb_params)
    
    # Process data with current parameter set
    glucose_data, combined_data = data
    processed_data = add_features_bezier(params, features, {patient: (glucose_data, combined_data)}, 
                                prediction_horizon, patient)

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        processed_data.drop(features_to_remove, axis=1), processed_data['glucose_next'], train_size=train_size, random_state=random_seed
    )
    
    # Train model
    local_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks
    )
    
    # Make predictions and calculate RMSE
    preds = local_model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, preds))

# Initial control points for Bezier curves
# Format: [x1, y1, x2, y2, x3, y3, x4, y4] - Four control points per feature
base_control_points = {
    'simple_sugars': [0.0, 0.0, 0.5, 0.8, 1.5, 0.5, 3.0, 0.0],       # Fast rise, quick drop
    'complex_sugars': [0.0, 0.0, 1.0, 0.5, 3.0, 0.7, 5.0, 0.0],      # Slower rise, longer effect
    'proteins': [0.0, 0.0, 2.0, 0.3, 5.0, 0.6, 9.0, 0.0],           # Very slow rise, extended effect
    'fats': [0.0, 0.0, 3.0, 0.2, 7.0, 0.4, 12.0, 0.0],              # Slowest rise, longest effect
    'dietary_fibers': [0.0, 0.0, 1.0, 0.1, 3.0, 0.3, 6.0, 0.0],      # Moderate effect curve
    'fast_insulin': [0.0, 0.0, 0.3, 0.9, 1.5, 0.3, 3.0, 0.0],        # Fast action, quick drop
    'slow_insulin': [0.0, 0.0, 1.0, 0.4, 4.0, 0.7, 8.0, 0.0]         # Slower action, extended effect
}

# Run optimization
def main(load_params=False):
    if load_params:
        try:
            with open('parameters/patient_bezier_params.pkl', 'rb') as f:
                patient_params = pickle.load(f)
            print("Loaded existing bezier parameters")
        except FileNotFoundError:
            print("No existing parameters found, running optimization")
            patient_params = optimize_patient_parameters_bezier(patients, 6, base_control_points)
            # Save the optimized parameters
            os.makedirs('parameters', exist_ok=True)
            with open('parameters/patient_bezier_params.pkl', 'wb') as f:
                pickle.dump(patient_params, f)
    else:
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
                test_days = days[3:]  
                
                for test_day in test_days:
                    day_mask = processed_data['datetime'].dt.day == test_day
                    test_day_data = processed_data[day_mask]
                    hours = sorted(test_day_data['hour'].unique())
                    
                    for hour in hours:
                        hour_mask = test_day_data['hour'] == hour
                        test = test_day_data[hour_mask]
                        
                        # Train on all data before the current hour from any day
                        earliest_test_time = test['datetime'].min()
                        safe_train_mask = processed_data['datetime'] < earliest_test_time
                        
                        train = processed_data[safe_train_mask]
                        
                        hour_model = lgb.LGBMRegressor(**lgb_params)
                        
                        X_train, X_val, y_train, y_val = train_test_split(
                            train.drop(features_to_remove, axis=1), train['glucose_next'], 
                            train_size=train_size, random_state=42
                        )
                        
                        hour_model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)], 
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

    df.to_csv('evaluation_metrics_bezier.csv', index=False) 

if __name__ == "__main__":
    main(load_params=True)  # Set to True to load existing parameters 