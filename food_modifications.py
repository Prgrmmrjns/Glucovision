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
prediction_horizon = 6  # Using the model from directory 12
model_path = f'models/pixtral-large-latest/{prediction_horizon}'
increments = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]  # Smaller increments to stay closer to training data range

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
    
    # Filter glucose data for 0-2 hours after each meal
    meal_times = combined_data['datetime'].unique()
    mask = np.zeros(len(glucose_data), dtype=bool)
    for meal_time in meal_times:
        time_diff = (glucose_data['datetime'] - meal_time).dt.total_seconds() / 3600
        mask |= (time_diff >= 0) & (time_diff <= 2)
    
    glucose_data = glucose_data[mask]
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

def analyze_nutrient_ranges(patients, meal_features):
    """Analyze the range and distribution of nutrients for each patient."""
    print("\n=== Nutrient Ranges for Each Patient ===")
    for patient in patients:
        print(f"\nPatient {patient}:")
        _, combined_data = get_data(patient, prediction_horizon)
        
        for feature in meal_features:
            # Get non-zero values only
            feature_values = combined_data[combined_data[feature] > 0][feature]
            if len(feature_values) > 0:
                print(f"  {feature}: {len(feature_values)} meals, range: {feature_values.min():.1f}-{feature_values.max():.1f}g, mean: {feature_values.mean():.1f}g, median: {feature_values.median():.1f}g")
            else:
                print(f"  {feature}: No meals with this nutrient")

# Load models
models, feature_names = load_models()

# Analyze nutrient ranges before making modifications
analyze_nutrient_ranges(patients, meal_features)

# Create DataFrame to store results
results_df = pd.DataFrame()


# Simulate additions/subtractions for each patient and feature
for patient in patients:
    model = models[patient]
    patient_params_subset = {k: v for k, v in patient_params[patient]['bezier_points'].items() if k in features}
    
    # Get original data (already filtered for 0-2 hours post-meal)
    data = get_data(patient, prediction_horizon)
    processed_data = add_features(patient_params_subset, features, data, prediction_horizon)
    X_test_orig = processed_data.drop(features_to_remove, axis=1)
    preds_orig = X_test_orig['glucose'] - model.predict(X_test_orig)
    baseline_glucose = np.mean(preds_orig)
    
    # Debug - confirm baseline is calculated on 0-2 hour window
    print(f"Baseline glucose (0-2h post-meal) for patient {patient}: {baseline_glucose:.2f}")
    
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
            mean_glucose = np.mean(preds_mod)
            
            # Store results
            results_df = pd.concat([results_df, pd.DataFrame({
                'feature': [feature],
                'increment': [increment],
                'mean_glucose': [mean_glucose],
                'patient': [patient]
            })])

# Create pivot table for visualization
pivot_df = pd.pivot_table(
    results_df,
    index='increment',
    columns=['feature', 'patient'],
    values=['mean_glucose']
)

# Calculate mean and standard error across patients for each feature and increment
metric_cols = ['mean_glucose']
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
        axes[i].set_xlabel('Decrement / Increment (g)', fontsize=8)
        axes[i].set_ylabel('Change in Mean Glucose (mg/dL)', fontsize=8)
        axes[i].grid(True)
        axes[i].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[i].text(-0.12, 1.05, letters[i], transform=axes[i].transAxes, fontsize=16, fontweight='bold', va='top')
    
    # Use the last subplot space for the legend
    axes[5].axis('off')  # Turn off axis for the legend subplot
    
    # Collect handles and labels from the last data subplot
    handles, labels = axes[4].get_legend_handles_labels()
    
    # Add legend to the empty subplot
    axes[5].legend(handles, labels, title='Patient', loc='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('images/results/all_features_mean_glucose.png', dpi=300)
    plt.savefig('images/results/all_features_mean_glucose.eps', dpi=300)
    plt.close()

# Only plot mean glucose changes across all features
plot_all_features_mean_glucose(results_df)

# Save results
# Transform to relative glucose values
relative_results_df = results_df.copy()
# For each patient and feature, calculate relative changes from baseline
for patient in patients:
    for feature in meal_features:
        # Get baseline (increment=0) glucose value
        baseline = results_df[(results_df['patient'] == patient) & 
                             (results_df['feature'] == feature) & 
                             (results_df['increment'] == 0)]['mean_glucose'].values[0]
        
        # Calculate relative changes
        mask = (relative_results_df['patient'] == patient) & (relative_results_df['feature'] == feature)
        relative_results_df.loc[mask, 'mean_glucose'] = relative_results_df.loc[mask, 'mean_glucose'] - baseline

# Save results with relative glucose values
relative_results_df.to_csv('analysis/food_modification_results.csv', index=False)