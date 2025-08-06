import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import warnings
from params import *
from processing_functions import *

warnings.filterwarnings('ignore')

# Script-specific constants
meal_features = ['simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers']
increments = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]
prediction_horizon = 6  # 30 minutes

# Redundant functions removed - using imports from processing_functions.py

def modify_macronutrients(combined_data, nutrient, amount):
    """Modify a specific macronutrient by a given amount for all meals."""
    modified_data = combined_data.copy()
    # Modify all food events (where any macronutrient is > 0)
    food_mask = (modified_data['simple_sugars'] > 0) | (modified_data['complex_sugars'] > 0) | \
                (modified_data['proteins'] > 0) | (modified_data['fats'] > 0) | (modified_data['dietary_fibers'] > 0)
    
    if food_mask.any():
        modified_data.loc[food_mask, nutrient] += amount
        # Ensure values remain non-negative
        modified_data[nutrient] = modified_data[nutrient].clip(lower=0)
    return modified_data

# Load global Bézier parameters from d1namo
with open(f'{RESULTS_PATH}/d1namo_bezier_params.json', 'r') as f:
    global_params = json.load(f)


# Store results
results_df = pd.DataFrame()

# For each patient, analyze effect of macronutrient modifications
for patient in PATIENTS_D1NAMO:
    print(f"Processing patient {patient}")
    
    # Get patient data
    glucose_data, combined_data = get_d1namo_data(patient)
    
    # Get baseline prediction using original data
    baseline_data = add_d1namo_features(global_params, OPTIMIZATION_FEATURES_D1NAMO, (glucose_data, combined_data))
    
    # Use only data from test days (4-5) for consistency with d1namo
    test_days = baseline_data['datetime'].dt.day.unique()[3:]
    baseline_test = baseline_data[baseline_data['datetime'].dt.day.isin(test_days)]

    # Prepare training data (use data from all patients like d1namo)
    all_train_data = []
    for train_patient in PATIENTS_D1NAMO:
        train_glucose, train_combined = get_d1namo_data(train_patient)
        train_processed = add_d1namo_features(global_params, OPTIMIZATION_FEATURES_D1NAMO, (train_glucose, train_combined))
        # Use only training days (1-3)
        train_days = train_processed['datetime'].dt.day.unique()[:3]
        train_subset = train_processed[train_processed['datetime'].dt.day.isin(train_days)]
        if len(train_subset) > 0:
            all_train_data.append(train_subset)
    

    # Combine all training data
    X_train_all = pd.concat(all_train_data, ignore_index=True)
    
    # Remove features not used for prediction
    available_features = X_train_all.columns.difference(FEATURES_TO_REMOVE_D1NAMO)
    X_train = X_train_all[available_features]
    y_train = X_train_all[f'glucose_{prediction_horizon}']
    
    # Train model
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    model = lgb.train(LGB_PARAMS, lgb.Dataset(X_tr, label=y_tr), 
                     valid_sets=[lgb.Dataset(X_val, label=y_val)])
    
    # Get baseline prediction
    X_baseline = baseline_test[available_features]
    baseline_predictions = model.predict(X_baseline)
    baseline_mean = np.mean(baseline_predictions)
    
    # For each macronutrient, test modifications
    for feature in meal_features:
        for increment in increments:
            # Modify the combined data
            modified_combined = modify_macronutrients(combined_data, feature, increment)
            
            # Re-generate features with modified data
            modified_data = add_d1namo_features(global_params, OPTIMIZATION_FEATURES_D1NAMO, (glucose_data, modified_combined))
            modified_test = modified_data[modified_data['datetime'].dt.day.isin(test_days)]
            
            if len(modified_test) == 0:
                continue
            
            # Make predictions with modified data
            X_modified = modified_test[available_features]
            modified_predictions = model.predict(X_modified)
            modified_mean = np.mean(modified_predictions)
            
            # Calculate change
            glucose_change = modified_mean - baseline_mean
            
            # Store results
            results_df = pd.concat([results_df, pd.DataFrame({
                'patient': [patient],
                'feature': [feature],
                'increment': [increment],
                'glucose_change': [glucose_change],
                'baseline_prediction': [baseline_mean],
                'modified_prediction': [modified_mean]
            })], ignore_index=True)

print(f"Analysis complete. Generated {len(results_df)} data points.")

# Create visualization
def create_macronutrient_modification_plot():
    """Create a comprehensive visualization of macronutrient modification effects"""
    
    # Calculate patient averages for each feature and increment
    avg_results = results_df.groupby(['feature', 'increment'])['glucose_change'].agg(['mean', 'std']).reset_index()
    
    # Create figure with subplots for each macronutrient
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Colors for patients
    patient_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    
    # Plot each macronutrient
    for i, feature in enumerate(meal_features):
        ax = axes[i]
        
        # Plot individual patients as thin lines
        for j, patient in enumerate(PATIENTS_D1NAMO):
            patient_data = results_df[(results_df['feature'] == feature) & (results_df['patient'] == patient)]
            if len(patient_data) > 0:
                patient_data = patient_data.sort_values('increment')
                ax.plot(patient_data['increment'], patient_data['glucose_change'], 
                       color=patient_colors[j], alpha=0.6, linewidth=2, label=f'Patient {patient}')
        
        # Plot average as thick line
        feature_avg = avg_results[avg_results['feature'] == feature].sort_values('increment')
        ax.plot(feature_avg['increment'], feature_avg['mean'], 
               color='black', linewidth=4, label='Average', marker='o', markersize=6)
        
        # Formatting
        ax.set_title(f'{feature.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Modification Amount (g)', fontsize=12)
        ax.set_ylabel('Change in Predicted Glucose (mg/dL)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add subplot label
        ax.text(0.02, 0.98, chr(65 + i), transform=ax.transAxes, fontsize=16, 
               fontweight='bold', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Use last subplot for legend
    axes[5].axis('off')
    
    # Create legend
    legend_elements = []
    for j, patient in enumerate(PATIENTS_D1NAMO):
        legend_elements.append(plt.Line2D([0], [0], color=patient_colors[j], alpha=0.8, linewidth=2, label=f'Patient {patient}'))
    legend_elements.append(plt.Line2D([0], [0], color='black', linewidth=4, label='Average'))
    
    axes[5].legend(handles=legend_elements, loc='center', fontsize=12, title='Patients', title_fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('../manuscript/images/results/macronutrient_modifications.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return avg_results

# Create visualization
avg_results = create_macronutrient_modification_plot()

# Generate summary statistics
print("\nSummary of Macronutrient Modification Effects:")
print("=" * 60)

for feature in meal_features:
    feature_data = avg_results[avg_results['feature'] == feature]
    baseline = feature_data[feature_data['increment'] == 0]['mean'].iloc[0]
    max_positive = feature_data[feature_data['increment'] == 50]['mean'].iloc[0]
    max_negative = feature_data[feature_data['increment'] == -50]['mean'].iloc[0]
    
    print(f"\n{feature.replace('_', ' ').title()}:")
    print(f"  +50g effect: {max_positive:+.2f} mg/dL")
    print(f"  -50g effect: {max_negative:+.2f} mg/dL")
    print(f"  Total range: {max_positive - max_negative:.2f} mg/dL")

# Save detailed results
results_df.to_csv(f'{RESULTS_PATH}/macronutrient_modification_results.csv', index=False)
avg_results.to_csv(f'{RESULTS_PATH}/macronutrient_modification_averages.csv', index=False)
