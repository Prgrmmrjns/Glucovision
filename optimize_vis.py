import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from scipy.special import comb
from sklearn.preprocessing import MinMaxScaler

# Create output directory
os.makedirs('optimization_plots', exist_ok=True)

# Constants
features = ['simple_sugars', 'complex_sugars', 'fats', 'dietary_fibers', 'proteins', 'insulin', 'calories']
patients = ['001', '002', '004', '006', '007', '008']

# Load the patient params
with open('parameters/patient_bezier_params.json', 'r') as f:
    patient_params = json.load(f)

# Define Bezier curve function (from main.py)
def bezier_curve(points, num=50):
    """Generate Bezier curve from control points using Bernstein polynomials"""
    n = len(points) - 1  # Degree of curve is n
    t = np.linspace(0, 1, num)
    curve = np.zeros((num, 2))
    
    for i, point in enumerate(points):
        # Calculate Bernstein polynomial basis
        curve += np.outer(comb(n, i) * (t**i) * ((1-t)**(n-i)), point)
    
    return curve[np.argsort(curve[:, 0])]

# Function to process glucose data
def get_glucose_data(patient, prediction_horizon=6):
    glucose_data = pd.read_csv(f"diabetes_subset_pictures-glucose-food-insulin/{patient}/glucose.csv")
    glucose_data["datetime"] = pd.to_datetime(glucose_data["date"] + ' ' + glucose_data["time"])
    glucose_data = glucose_data.drop(['type', 'comments', 'date', 'time'], axis=1)
    glucose_data['glucose'] *= 18.0182
    glucose_data['hour'] = glucose_data['datetime'].dt.hour
    glucose_data['time'] = glucose_data['hour'] + glucose_data['datetime'].dt.minute / 60
    
    # Calculate target variable
    glucose_data['glucose_next'] = glucose_data['glucose'] - glucose_data['glucose'].shift(-prediction_horizon)
    glucose_data.dropna(subset=['glucose_next'], inplace=True)
    
    return glucose_data

# Function to process feature data
def get_feature_data(patient):
    insulin_data = pd.read_csv(f"diabetes_subset_pictures-glucose-food-insulin/{patient}/insulin.csv")
    insulin_data["datetime"] = pd.to_datetime(insulin_data["date"] + ' ' + insulin_data["time"])
    insulin_data['insulin'] = insulin_data['slow_insulin'] + insulin_data['fast_insulin']
    insulin_data = insulin_data.drop(['slow_insulin', 'fast_insulin', 'comment', 'date', 'time'], axis=1)
    
    food_data = pd.read_csv(f"food_data/pixtral-large-latest/{patient}.csv")
    food_data['datetime'] = pd.to_datetime(food_data['datetime'], format='%Y:%m:%d %H:%M:%S')
    food_data = food_data[['datetime', 'simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers', 'calories']]
    
    combined_data = pd.concat([food_data, insulin_data]).sort_values('datetime').reset_index(drop=True)
    combined_data.fillna(0, inplace=True)
    
    return combined_data

# Function to calculate feature impacts using Bezier curves
def calculate_feature_impacts(glucose_data, feature_data, bezier_params, weights, prediction_horizon=6):
    # Convert datetime to nanoseconds for efficient vectorized operations
    glucose_times = glucose_data['datetime'].values.astype('datetime64[s]').astype(np.int64)
    feature_times = feature_data['datetime'].values.astype('datetime64[s]').astype(np.int64)
    
    # Calculate time difference matrix (in hours)
    time_diff_hours = ((glucose_times[:, None] - feature_times[None, :]) / 3600)
    
    # DataFrame to hold results
    results = pd.DataFrame(index=glucose_data.index)
    
    # Calculate impact for each feature
    for feature in features:
        if feature in bezier_params and feature in weights:
            # Generate Bezier curve
            points = np.array(bezier_params[feature]).reshape(-1, 2)
            curve = bezier_curve(points, num=100)
            x_curve, y_curve = curve[:, 0], curve[:, 1]
            
            # Create weights array
            curve_weights = np.zeros_like(time_diff_hours)
            
            # For each time difference, find the closest point on bezier curve
            for i in range(len(glucose_times)):
                for j in range(len(feature_times)):
                    if time_diff_hours[i, j] >= 0 and time_diff_hours[i, j] <= max(x_curve):
                        # Find closest x value in curve
                        idx = np.abs(x_curve - time_diff_hours[i, j]).argmin()
                        curve_weights[i, j] = y_curve[idx]
            
            # Calculate feature impact
            feature_values = pd.Series(np.dot(curve_weights, feature_data[feature].values), index=glucose_data.index)
            shifted_feature_values = feature_values.shift(-prediction_horizon)
            results[feature] = shifted_feature_values - feature_values
    
    return results

# Function to calculate weighted sum of feature impacts
def calculate_weighted_sum(feature_impacts, weights):
    weighted_sum = pd.Series(0, index=feature_impacts.index)
    for feature in weights:
        if feature in feature_impacts.columns:
            weighted_sum += feature_impacts[feature] * weights[feature]
    return weighted_sum

# Process each patient
patient = '001'
    
# Get data
glucose_data = get_glucose_data(patient)
feature_data = get_feature_data(patient)

# Get patient-specific parameters for "pixtral-large-latest"
bezier_params = patient_params[patient]['bezier_points']
weights = patient_params[patient]['weights']

# Filter for first three days only
first_days = glucose_data['datetime'].dt.day.unique()[:3]
glucose_data = glucose_data[glucose_data['datetime'].dt.day.isin(first_days)]

# Calculate feature impacts
feature_impacts = calculate_feature_impacts(glucose_data, feature_data, bezier_params, weights)

# Calculate weighted sum
weighted_sum = calculate_weighted_sum(feature_impacts, weights)

# Create a figure
plt.figure(figsize=(14, 6))

# Normalize data for visualization
scaler = MinMaxScaler()
glucose_next_scaled = scaler.fit_transform(glucose_data[['glucose_next']]).flatten()

# Check if weighted_sum has any non-zero values
if weighted_sum.abs().sum() > 0:
    weighted_sum_scaled = scaler.fit_transform(weighted_sum.values.reshape(-1, 1)).flatten()
else:
    weighted_sum_scaled = np.zeros_like(glucose_next_scaled)

# Plot data
plt.plot(glucose_data['datetime'], glucose_next_scaled, 'b-', linewidth=2, label='Glucose')
plt.plot(glucose_data['datetime'], weighted_sum_scaled, 'r-', linewidth=2, alpha=0.7, label='Weighted sum of \nmacronutrients and insulin')

# Customize plot

# Plot individual feature impacts with low opacity
colors = ['g', 'c', 'm', 'y', 'orange', 'purple', 'brown']
for i, feature in enumerate(features):
    if feature in feature_impacts.columns and feature_impacts[feature].abs().sum() > 0:
        feature_scaled = scaler.fit_transform(feature_impacts[feature].values.reshape(-1, 1)).flatten()
        plt.plot(glucose_data['datetime'], feature_scaled, color=colors[i % len(colors)], 
                    alpha=0.3, linewidth=1, label=feature)

# Add legend inside the plot
plt.legend(loc='upper right', bbox_to_anchor=(1.18, 1))

# Add axis labels
plt.ylabel('Normalized Feature Values')
plt.xlabel('Time')

# Adjust layout
plt.tight_layout()

# Save plot
plt.savefig(f'images/supplementary_data/patient_{patient}_optimization.png')
plt.close()