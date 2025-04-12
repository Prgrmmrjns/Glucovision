import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle
from pathlib import Path
from scipy.special import comb
import warnings

# Constants
patients = ['001', '002', '004', '006', '007', '008']
meal_features = ['simple_sugars', 'complex_sugars', 'fats', 'dietary_fibers', 'proteins']
features = meal_features + ['insulin']
features_to_remove = ['glucose_next', 'datetime', 'hour']
prediction_horizon = 6  # Using the model from directory 12
model_path = f'models/pixtral-large-latest/{prediction_horizon}'

warnings.filterwarnings('ignore')

# Load Bezier parameters
with open('parameters/patient_bezier_params.json', 'r') as f:
    patient_params = json.load(f)

# Load models
def load_models():
    models = {}
    feature_names = {}
    for patient in patients:
        with open(Path(model_path) / f'patient_{patient}_model.pkl', 'rb') as f:
            model = pickle.load(f)
            models[patient] = model
            feature_names[patient] = model.feature_name_
    return models, feature_names

def bezier_curve(points, num=50):
    """Generate Bezier curve from control points using Bernstein polynomials"""
    n = len(points) - 1
    t = np.linspace(0, 1, num)
    curve = np.zeros((num, 2))
    
    for i, point in enumerate(points):
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
        mask = (time_diff_hours >= 0) & (time_diff_hours <= max(x_curve))
        for i, j in zip(*np.where(mask)):
            t_diff = time_diff_hours[i, j]
            idx = np.abs(x_curve - t_diff).argmin()
            weights[i, j] = y_curve[idx]
        
        # Compute impact and shift by prediction horizon
        feature_values = pd.Series(np.dot(weights, combined_data[feature].values))
        glucose_data[feature] = feature_values.shift(-prediction_horizon) - feature_values
    
    return glucose_data

# Analyze time-of-day effect on glucose
def modify_time(data, target_hour):
    """Modify the time of day for all data points while preserving date."""
    glucose_data, combined_data = data
    
    # Create deep copies to avoid modifying originals
    modified_glucose = glucose_data.copy()
    modified_combined = combined_data.copy()
    
    # For glucose data, set the hour component while preserving the date
    original_dates = modified_glucose['datetime'].dt.date
    original_minutes = modified_glucose['datetime'].dt.minute
    modified_glucose['datetime'] = original_dates.apply(
        lambda date: pd.Timestamp(date.year, date.month, date.day, target_hour, 0)
    )
    # Update hour and time fields
    modified_glucose['hour'] = target_hour
    modified_glucose['time'] = target_hour + original_minutes / 60
    
    # For combined data (food and insulin), set the hour component while preserving the date
    original_combined_dates = modified_combined['datetime'].dt.date
    modified_combined['datetime'] = original_combined_dates.apply(
        lambda date: pd.Timestamp(date.year, date.month, date.day, target_hour, 0)
    )
    
    return modified_glucose, modified_combined

# Load models
models, feature_names = load_models()


def analyze_time_effect():
    """Analyze the effect of time of day on glucose predictions."""
    time_results = pd.DataFrame()
    
    for patient in patients:
        model = models[patient]
        patient_params_subset = {k: v for k, v in patient_params[patient]['bezier_points'].items() if k in features}
        
        # Get original data for this patient
        original_data = get_data(patient, prediction_horizon)
        glucose_data, _ = original_data
        
        # Run predictions for each hour of the day
        for hour in range(24):
            # Modify the time to the current hour
            modified_data = modify_time(original_data, hour)
            processed_data = add_features(patient_params_subset, features, modified_data, prediction_horizon)
            
            # Make predictions
            X_test = processed_data.drop(features_to_remove, axis=1)
            preds = X_test['glucose'] - model.predict(X_test)
            
            # Calculate metrics
            time_results = pd.concat([time_results, pd.DataFrame({
                'hour': [hour] * len(preds),
                'glucose_prediction': preds,
                'patient': [patient] * len(preds)
            })])
    
    return time_results

def plot_time_effect(time_df):
    """Plot the effect of time of day on glucose predictions."""
    plt.figure(figsize=(15, 5))
    
    # Calculate average glucose by hour and patient
    hour_avg = time_df.groupby(['patient', 'hour'])['glucose_prediction'].mean().reset_index()
    
    # Calculate baseline (average across all hours) for each patient
    patient_baselines = hour_avg.groupby('patient')['glucose_prediction'].mean().to_dict()
    
    # Plot relative change from baseline for each patient
    for patient in patients:
        patient_data = hour_avg[hour_avg['patient'] == patient]
        baseline = patient_baselines[patient]
        
        plt.plot(
            patient_data['hour'], 
            patient_data['glucose_prediction'] - baseline,
            marker='o', 
            linewidth=2,
            label=f'Patient {patient}'
        )
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xticks(range(24))
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Relative Change in Predicted Glucose (mg/dL)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figures
    plt.savefig('images/supplementary_data/time_effect_glucose_predictions.png', dpi=300)
    plt.savefig('images/supplementary_data/time_effect_glucose_predictions.eps', dpi=300)
    plt.close()

time_df = analyze_time_effect()

# Interpret time effect findings
def interpret_time_effect(time_df):
    """Analyze and interpret the time-of-day effect on glucose predictions."""
    
    # Calculate average glucose by hour and patient
    hour_avg = time_df.groupby(['patient', 'hour'])['glucose_prediction'].mean().reset_index()
    
    # Calculate baseline (average across all hours) for each patient
    patient_baselines = hour_avg.groupby('patient')['glucose_prediction'].mean().to_dict()
    
    # Calculate relative change from baseline
    hour_avg['relative_change'] = hour_avg.apply(
        lambda row: row['glucose_prediction'] - patient_baselines[row['patient']], 
        axis=1
    )
    
    # Find peak hours and lowest hours for each patient
    summary = pd.DataFrame(columns=['patient', 'peak_hour', 'peak_change', 'lowest_hour', 'lowest_change', 'range'])
    
    for patient in patients:
        patient_data = hour_avg[hour_avg['patient'] == patient]
        
        peak_row = patient_data.loc[patient_data['relative_change'].idxmax()]
        lowest_row = patient_data.loc[patient_data['relative_change'].idxmin()]
        
        peak_hour = peak_row['hour']
        peak_change = peak_row['relative_change']
        
        lowest_hour = lowest_row['hour']
        lowest_change = lowest_row['relative_change']
        
        glucose_range = peak_change - lowest_change
        
        summary = pd.concat([summary, pd.DataFrame({
            'patient': [patient],
            'peak_hour': [peak_hour],
            'peak_change': [peak_change],
            'lowest_hour': [lowest_hour],
            'lowest_change': [lowest_change],
            'range': [glucose_range]
        })], ignore_index=True)
    
    # Calculate morning (6-12), afternoon (12-18), evening (18-0), night (0-6) averages
    time_periods = {
        'morning': range(6, 12),
        'afternoon': range(12, 18),
        'evening': range(18, 24),
        'night': range(0, 6)
    }
    
    period_avgs = pd.DataFrame(columns=['patient', 'period', 'avg_change'])
    
    for patient in patients:
        patient_data = hour_avg[hour_avg['patient'] == patient]
        
        for period, hours in time_periods.items():
            period_data = patient_data[patient_data['hour'].isin(hours)]
            avg_change = period_data['relative_change'].mean()
            
            period_avgs = pd.concat([period_avgs, pd.DataFrame({
                'patient': [patient],
                'period': [period],
                'avg_change': [avg_change]
            })], ignore_index=True)
    
    # Find common patterns
    common_patterns = {}
    common_patterns['avg_peak_hour'] = summary['peak_hour'].mean()
    common_patterns['avg_lowest_hour'] = summary['lowest_hour'].mean()
    common_patterns['avg_glucose_range'] = summary['range'].mean()
    
    # Calculate overall period effect
    overall_period_effect = period_avgs.groupby('period')['avg_change'].mean().reset_index()
    common_patterns['period_ranking'] = overall_period_effect.sort_values('avg_change', ascending=False)['period'].tolist()
    
    # Combine all results into a single DataFrame
    all_results = pd.DataFrame()
    
    # Add summary data with result_type
    summary['result_type'] = 'patient_summary'
    all_results = pd.concat([all_results, summary])
    
    # Add period averages with result_type
    period_avgs['result_type'] = 'period_average'
    all_results = pd.concat([all_results, period_avgs])
    
    # Add common patterns
    common_patterns_df = pd.DataFrame({
        'result_type': ['common_pattern'] * 3,
        'metric': ['avg_peak_hour', 'avg_lowest_hour', 'avg_glucose_range'],
        'value': [common_patterns['avg_peak_hour'], common_patterns['avg_lowest_hour'], common_patterns['avg_glucose_range']]
    })
    all_results = pd.concat([all_results, common_patterns_df])
    
    # Add period rankings
    for i, period in enumerate(common_patterns['period_ranking']):
        period_rank_df = pd.DataFrame({
            'result_type': ['period_ranking'],
            'period': [period],
            'rank': [i+1]
        })
        all_results = pd.concat([all_results, period_rank_df])
    
    # Add hour averages
    hour_avg['result_type'] = 'hour_average'
    all_results = pd.concat([all_results, hour_avg])
    
    return all_results

# Get interpretation results and save to a single CSV file
all_results = interpret_time_effect(time_df)
all_results.to_csv('analysis/time_effect_results.csv', index=False)

# Plot time effect
plot_time_effect(time_df)

# Print interpretation summary
print("\nTime Effect Analysis Results:")
print("-" * 50)

# Get summary statistics
summary_data = all_results[all_results['result_type'] == 'patient_summary']
common_patterns = all_results[all_results['result_type'] == 'common_pattern']
period_rankings = all_results[all_results['result_type'] == 'period_ranking'].sort_values('rank')

avg_peak_hour = float(common_patterns[common_patterns['metric'] == 'avg_peak_hour']['value'])
avg_lowest_hour = float(common_patterns[common_patterns['metric'] == 'avg_lowest_hour']['value'])
avg_glucose_range = float(common_patterns[common_patterns['metric'] == 'avg_glucose_range']['value'])

print(f"Average peak glucose hour: {avg_peak_hour:.1f}")
print(f"Average lowest glucose hour: {avg_lowest_hour:.1f}")
print(f"Average glucose range due to time effect: {avg_glucose_range:.1f} mg/dL")

print("\nPeriod ranking from highest to lowest glucose effect:")
for _, row in period_rankings.iterrows():
    period_avgs = all_results[(all_results['result_type'] == 'period_average') & (all_results['period'] == row['period'])]
    avg = period_avgs['avg_change'].mean()
    print(f"{int(row['rank'])}. {row['period'].capitalize()}: {avg:.1f} mg/dL")

print("\nPatient-specific peak and lowest hours:")
for _, row in summary_data.iterrows():
    print(f"Patient {row['patient']}: Peak at hour {int(row['peak_hour'])} ({row['peak_change']:.1f} mg/dL), Lowest at hour {int(row['lowest_hour'])} ({row['lowest_change']:.1f} mg/dL)")

# Analysis complete message
print("\nTime effect analysis complete. Results saved to analysis/time_effect_results.csv")