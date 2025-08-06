import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import json
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import warnings
from params import *
from processing_functions import *

warnings.filterwarnings('ignore')


# Load Bézier parameters
with open(f'{RESULTS_PATH}/d1namo_bezier_params.json', 'r') as f:
    global_params = json.load(f)

# Load data from patient 008 (as in original script)
patient = '008'
glucose_data, combined_data = get_d1namo_data(patient)

# Add macronutrient features using Bézier curves
patient_data = add_d1namo_features(global_params, OPTIMIZATION_FEATURES_D1NAMO, (glucose_data, combined_data))

# Create output directory
os.makedirs('../manuscript/images/graphical_abstract', exist_ok=True)

# Select time period for visualization (3 days of test data)
test_days = patient_data['datetime'].dt.day.unique()[3:6]  # Last 3 days
plot_data = patient_data[patient_data['datetime'].dt.day.isin(test_days)].copy()

# Limit to a subset for cleaner visualization (every 4th point)
plot_data = plot_data.iloc[::4].reset_index(drop=True)

# Calculate sum of modeled features
feature_sum = plot_data[OPTIMIZATION_FEATURES_D1NAMO].sum(axis=1)

# Scale both glucose and feature sum
scaler = MinMaxScaler()
glucose_scaled = scaler.fit_transform(plot_data[['glucose']])
feature_sum_scaled = scaler.fit_transform(feature_sum.values.reshape(-1, 1))

# Plot: Individual macronutrient features
plt.figure(figsize=(14, 4))

# Colors for consistency with other plots
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
feature_names = ['Simple Sugars', 'Complex Sugars', 'Proteins', 'Fats', 'Dietary Fibers', 'Insulin']

# Plot 1: Glucose only
plt.figure(figsize=(12, 4))
plt.plot(plot_data['datetime'], glucose_scaled, color='black', linewidth=3, label='Glucose')

# Minimalistic styling
plt.xticks([])
plt.yticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

# Simple legend
plt.legend(loc='upper right', fontsize=14, frameon=False)
plt.tight_layout()
plt.savefig('../manuscript/images/graphical_abstract/glucose_only.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Individual macronutrient features
plt.figure(figsize=(14, 4))

# Scale individual features
scaler_features = MinMaxScaler()
features_scaled = scaler_features.fit_transform(plot_data[OPTIMIZATION_FEATURES_D1NAMO])

# Plot glucose
plt.plot(plot_data['datetime'], glucose_scaled, 'k-', linewidth=4, label='Glucose', alpha=0.9)

# Plot individual features
for i, feature in enumerate(OPTIMIZATION_FEATURES_D1NAMO):
    plt.plot(plot_data['datetime'], features_scaled[:, i], color=colors[i], 
             linewidth=2, alpha=0.7, label=feature_names[i])

# Minimalistic styling
plt.xticks([])
plt.yticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

# Legend with all individual features
legend_labels = ['Glucose'] + feature_names
plt.legend(legend_labels, loc='upper right', fontsize=10, frameon=False)
plt.tight_layout()
plt.savefig('../manuscript/images/graphical_abstract/glucose_with_individual_features.png', dpi=300, bbox_inches='tight')
plt.close()

# Generate predictions using d1namo logic
prediction_horizon = DEFAULT_PREDICTION_HORIZON  # 60 minutes
target_feature = f'glucose_{prediction_horizon}'
features_to_remove_ph = FEATURES_TO_REMOVE_D1NAMO + [f'glucose_{h}' for h in PREDICTION_HORIZONS]
current_patient_weight = 10
validation_size = 0.2

# Load all patient data
all_data_list = []
for p in PATIENTS_D1NAMO:
    patient_glucose, patient_combined = get_d1namo_data(p)
    patient_data = add_d1namo_features(global_params, OPTIMIZATION_FEATURES_D1NAMO, (patient_glucose, patient_combined))
    patient_data['patient_id'] = f"patient_{p}"
    all_data_list.append(patient_data)
all_data = pd.concat(all_data_list, ignore_index=True)

# Focus on patient 001 for predictions
patient = '001'
patient_mask = all_data['patient_id'] == f"patient_{patient}"
test_days = all_data[patient_mask]['datetime'].dt.day.unique()[3:6]  # Last 3 days

predictions_list = []
ground_truth_list = []
timestamps_list = []

for test_day in test_days:
    day_hours = all_data[patient_mask & (all_data['datetime'].dt.day == test_day)]['hour'].unique()
    for hour in day_hours:
        test = all_data[patient_mask & (all_data['datetime'].dt.day == test_day) & (all_data['hour'] == hour)]
        if len(test) == 0:
            continue
            
        # Training data: all data before test time + other patients
        X = pd.concat([
            all_data[patient_mask & (all_data['datetime'] < test['datetime'].min())], 
            all_data[~patient_mask]
        ])
        
        if len(X) < 100:  # Skip if not enough training data
            continue
            
        # Prepare training data
        indices = train_test_split(range(len(X)), test_size=validation_size, random_state=42)
        weights = [np.where(X['patient_id'].values[idx] == f"patient_{patient}", current_patient_weight, 1) for idx in indices]
        available_features = X.columns.difference(features_to_remove_ph)
        
        train = X[available_features]
        X_train, y_train, weights_train = train.values[indices[0]], X[target_feature].values[indices[0]], weights[0]
        X_val, y_val, weights_val = train.values[indices[1]], X[target_feature].values[indices[1]], weights[1]
        
        # Train model
        model = lgb.train(
            LGB_PARAMS, 
            lgb.Dataset(X_train, label=y_train, weight=weights_train),
            valid_sets=[lgb.Dataset(X_val, label=y_val, weight=weights_val)]
        )
        
        # Make predictions
        test_features = test[available_features]
        predictions = model.predict(test_features.values)
        ground_truth = test[target_feature].values
        
        # Store results
        for i, (pred, gt, timestamp) in enumerate(zip(predictions, ground_truth, test['datetime'])):
            predictions_list.append(pred + test['glucose'].iloc[i])  # Add back current glucose
            ground_truth_list.append(gt + test['glucose'].iloc[i])   # Add back current glucose
            timestamps_list.append(timestamp)

# Create prediction plot
pred_df = pd.DataFrame({
    'datetime': timestamps_list,
    'predictions': predictions_list,
    'ground_truth': ground_truth_list
}).sort_values('datetime').reset_index(drop=True)

# Subsample for cleaner visualization
pred_df = pred_df.iloc[::3].reset_index(drop=True)

# Scale for visualization
scaler_pred = MinMaxScaler()
pred_scaled = scaler_pred.fit_transform(pred_df[['predictions', 'ground_truth']])

# Simple prediction plot
plt.figure(figsize=(14, 4))
plt.plot(pred_df['datetime'], pred_scaled[:, 1], 'k-', linewidth=4, alpha=0.9, label='Ground Truth')
plt.plot(pred_df['datetime'], pred_scaled[:, 0], 'r-', linewidth=3, alpha=0.8, label='Predictions')

# Minimalistic styling
plt.xticks([])
plt.yticks([])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

# Simple legend
plt.legend(loc='upper right', fontsize=10, frameon=False)
plt.tight_layout()
plt.savefig('../manuscript/images/graphical_abstract/predictions_vs_ground_truth.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Generated predictions plot with {len(pred_df)} data points")