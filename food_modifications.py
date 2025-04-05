import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle
from pathlib import Path
from scipy.special import comb
import seaborn as sns
import warnings

# Constants
patients = ['001', '002', '004', '006', '007', '008']
meal_features = ['simple_sugars', 'complex_sugars', 'fats', 'dietary_fibers', 'proteins']
features = meal_features + ['insulin']
features_to_remove = ['glucose_next', 'datetime', 'hour']
prediction_horizon = 12  # Using the model from directory 12
model_path = f'models/pixtral-large-latest/{prediction_horizon}'
increments = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]

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

def modify_macronutrients(food_data, nutrient, amount):
    """Modify a specific macronutrient by a given amount."""
    modified_data = food_data.copy()
    modified_data[nutrient] += amount
    # Ensure values remain non-negative
    modified_data[nutrient] = modified_data[nutrient].clip(lower=0)
    return modified_data

# Load models
models, feature_names = load_models()

# Create DataFrame to store results
results_df = pd.DataFrame()

# Simulate additions/subtractions for each patient and feature
for patient in patients:
    model = models[patient]
    patient_params_subset = {k: v for k, v in patient_params[patient].items() if k in features}
    
    # Get original data
    data = get_data(patient, prediction_horizon)
    processed_data = add_features(patient_params_subset, features, data, prediction_horizon)
    X_test_orig = processed_data.drop(features_to_remove, axis=1)
    preds_orig = X_test_orig['glucose'] - model.predict(X_test_orig)
    baseline_glucose = np.mean(preds_orig)
    
    # Calculate hyper/hypo minutes in baseline
    baseline_hyper = np.sum(preds_orig > 180)
    baseline_hypo = np.sum(preds_orig < 70)
    
    # Test each feature modification
    for feature in meal_features:
        for increment in increments:
            # Get original food data
            glucose_data, original_combined = data
            food_df = original_combined[original_combined[feature] > 0].copy()
            
            # Modify food data
            food_df = modify_macronutrients(food_df, feature, increment)
            
            # Update combined data with modified values
            modified_combined = original_combined.copy()
            modified_combined.update(food_df)
            
            # Process with modified data
            modified_data = (glucose_data.copy(), modified_combined)
            processed_modified = add_features(patient_params_subset, features, modified_data, prediction_horizon)
            
            # Make predictions
            X_test_mod = processed_modified.drop(features_to_remove, axis=1)
            preds_mod = X_test_mod['glucose'] - model.predict(X_test_mod)
            
            # Calculate metrics
            hyper_minutes = np.sum(preds_mod > 180)
            hypo_minutes = np.sum(preds_mod < 70) 
            mean_glucose = np.mean(preds_mod)
            
            # Store results
            results_df = pd.concat([results_df, pd.DataFrame({
                'feature': [feature],
                'increment': [increment],
                'hyper_minutes': [hyper_minutes],
                'hypo_minutes': [hypo_minutes],
                'mean_glucose': [mean_glucose],
                'patient': [patient]
            })])

# Create pivot table for visualization
pivot_df = pd.pivot_table(
    results_df,
    index='increment',
    columns=['feature', 'patient'],
    values=['hyper_minutes', 'hypo_minutes', 'mean_glucose']
)

# Calculate mean and standard error across patients for each feature and increment
metric_cols = ['hyper_minutes', 'hypo_minutes', 'mean_glucose']
agg_df = results_df.groupby(['feature', 'increment'])[metric_cols].agg(['mean', 'std']).reset_index()

# Reshape for easier plotting
patient_avg = pd.DataFrame()
for metric in metric_cols:
    metric_data = agg_df[[(metric, 'mean'), (metric, 'std')]]
    metric_data.columns = ['mean', 'std']
    metric_data['metric'] = metric
    metric_data['feature'] = agg_df['feature']
    metric_data['increment'] = agg_df['increment']
    patient_avg = pd.concat([patient_avg, metric_data])

# Get baseline values for relative change calculation
baseline = patient_avg[patient_avg['increment'] == 0].copy()
baseline = baseline.set_index(['feature', 'metric'])['mean'].to_dict()

def plot_all_features_mean_glucose(results_df):
    """Plot mean glucose changes for all features in a single figure with 5 subplots."""
    # Create figure with 3x2 grid (5 subplots + 1 for legend)
    fig, axes = plt.subplots(2, 3, figsize=(16, 6))
    axes = axes.flatten()
    
    # Define letters for subplots
    letters = ['A', 'B', 'C', 'D', 'E']
    
    # Plot each feature in its own subplot
    for i, feature in enumerate(meal_features):
        # Filter data for this feature
        feature_data = results_df[results_df['feature'] == feature].copy()
        
        # Get baseline values for this feature
        baseline_vals = {}
        for patient in patients:
            patient_baseline = feature_data[(feature_data['patient'] == patient) & 
                                          (feature_data['increment'] == 0)]
            baseline_vals[patient] = patient_baseline['mean_glucose'].values[0]
        
        # Plot for each patient
        for patient in patients:
            patient_data = feature_data[feature_data['patient'] == patient]
            patient_data = patient_data.sort_values('increment')
            
            # Calculate relative change from baseline
            baseline_value = baseline_vals.get(patient, 0)
            relative_values = patient_data['mean_glucose'] - baseline_value
            
            axes[i].plot(
                patient_data['increment'], 
                relative_values,
                marker='o',
                label=patient
            )
        
        # Customize plot
        axes[i].set_title(feature)
        axes[i].set_xlabel('Increment')
        axes[i].set_ylabel('Change in Mean Glucose')
        axes[i].grid(True)
        axes[i].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[i].text(-0.1, 1.05, letters[i], transform=axes[i].transAxes, fontsize=16, fontweight='bold', va='top')
    
    # Use the last subplot space for the legend
    axes[5].axis('off')  # Turn off axis for the legend subplot
    
    # Collect handles and labels from the last data subplot
    handles, labels = axes[4].get_legend_handles_labels()
    
    # Add legend to the empty subplot
    axes[5].legend(handles, labels, title='Patient', loc='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('images/all_features_mean_glucose.png', dpi=300)
    plt.savefig('images/all_features_mean_glucose.eps', dpi=300)
    plt.close()

# Only plot mean glucose changes across all features
plot_all_features_mean_glucose(results_df)

# Save results
results_df.to_csv('analysis/food_modification_results.csv', index=False)

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

def analyze_time_effect():
    """Analyze the effect of time of day on glucose predictions."""
    time_results = pd.DataFrame()
    
    for patient in patients:
        model = models[patient]
        patient_params_subset = {k: v for k, v in patient_params[patient].items() if k in features}
        
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

# Save time analysis results
time_df.to_csv('analysis/time_effect_results.csv', index=False)

# Plot time effect
plot_time_effect(time_df)

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
    
    # Save summary to CSV
    summary.to_csv('analysis/time_effect_summary.csv', index=False)
    period_avgs.to_csv('analysis/time_effect_periods.csv', index=False)
    
    # Find common patterns
    common_patterns = {}
    common_patterns['avg_peak_hour'] = summary['peak_hour'].mean()
    common_patterns['avg_lowest_hour'] = summary['lowest_hour'].mean()
    common_patterns['avg_glucose_range'] = summary['range'].mean()
    
    # Calculate overall period effect
    overall_period_effect = period_avgs.groupby('period')['avg_change'].mean().reset_index()
    common_patterns['period_ranking'] = overall_period_effect.sort_values('avg_change', ascending=False)['period'].tolist()
    
    return summary, period_avgs, common_patterns

# Get interpretation results
summary, period_avgs, common_patterns = interpret_time_effect(time_df)

# Print interpretation results
print("\nTime Effect Analysis Results:")
print("-" * 50)
print(f"Average peak glucose hour: {common_patterns['avg_peak_hour']:.1f}")
print(f"Average lowest glucose hour: {common_patterns['avg_lowest_hour']:.1f}")
print(f"Average glucose range due to time effect: {common_patterns['avg_glucose_range']:.1f} mg/dL")
print("\nPeriod ranking from highest to lowest glucose effect:")
for i, period in enumerate(common_patterns['period_ranking']):
    avg = period_avgs[period_avgs['period'] == period]['avg_change'].mean()
    print(f"{i+1}. {period.capitalize()}: {avg:.1f} mg/dL")

print("\nPatient-specific peak and lowest hours:")
for _, row in summary.iterrows():
    print(f"Patient {row['patient']}: Peak at hour {int(row['peak_hour'])} ({row['peak_change']:.1f} mg/dL), Lowest at hour {int(row['lowest_hour'])} ({row['lowest_change']:.1f} mg/dL)")

# Analysis complete message
print("\nTime effect analysis complete. Results saved to analysis folder.")