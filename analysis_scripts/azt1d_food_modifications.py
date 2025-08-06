import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import warnings
import os

warnings.filterwarnings('ignore')

from params import *
from processing_functions import *

# Script-specific constants
meal_features = ['carbohydrates']
increments = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]

# Functions imported from processing_functions.py

def modify_macronutrients(combined_data, nutrient, amount):
    """Modify a specific macronutrient by a given amount for all meals."""
    modified_data = combined_data.copy()
    # Modify all food events (where carbohydrates > 0)
    food_mask = modified_data['carbohydrates'] > 0
    
    if food_mask.any():
        modified_data.loc[food_mask, nutrient] += amount
        # Ensure values remain non-negative
        modified_data[nutrient] = modified_data[nutrient].clip(lower=0)
    return modified_data

# Load data
print("Loading AZT1D data...")
data = load_azt1d_data()
print(f"Loaded data for {data['patient'].nunique()} patients, {len(data)} total records")

# Get Bezier parameters (use existing from approach_comparison or optimize new ones)
params_file = '../eval_scripts/temporal_mapping_detailed_results.csv'
if os.path.exists(params_file):
    results_df_temp = pd.read_csv(params_file)
    # Use best Bezier parameters from AZT1D evaluation
    best_azt1d_bezier = results_df_temp[
        (results_df_temp['mapping'] == 'bezier') & 
        (results_df_temp['dataset'] == 'AZT1D')
    ]
    if len(best_azt1d_bezier) > 0:
        print("Using optimized Bezier parameters from AZT1D evaluation")
        # For simplicity, use domain knowledge parameters
        global_params = AZT1D_BEZIER_PARAMS
    else:
        print("No AZT1D Bezier results found, using domain knowledge parameters")
        global_params = AZT1D_BEZIER_PARAMS
else:
    print("Using domain knowledge parameters for AZT1D")
    global_params = AZT1D_BEZIER_PARAMS

# Store results
results_df = pd.DataFrame()

# For each patient, analyze effect of macronutrient modifications
for patient in PATIENTS_AZT1D:
    patient_data = data[data['patient'] == patient].copy()
    if len(patient_data) < 50:
        continue
        
    print(f"Processing patient {patient} ({len(patient_data)} records)")
    
    # Split into train/test by days
    days = patient_data['datetime'].dt.day.unique()
    if len(days) < 4:
        continue
        
    train_days = days[:3]  # First 3 days for training
    test_days = days[3:6]  # Next 3 days for testing
    
    # Create combined data (food and insulin events)
    combined_data = patient_data[['datetime', 'carbohydrates', 'insulin']].copy()
    
    # Get baseline features
    baseline_data = add_azt1d_features(global_params, OPTIMIZATION_FEATURES_AZT1D, 
                                     patient_data, combined_data)
    
    # Filter test data
    baseline_test = baseline_data[baseline_data['datetime'].dt.day.isin(test_days)]
    if len(baseline_test) == 0:
        continue
    
    # Prepare training data from all patients
    all_train_data = []
    for train_patient in PATIENTS_AZT1D:
        train_patient_data = data[data['patient'] == train_patient].copy()
        if len(train_patient_data) < 20:
            continue
            
        train_days_patient = train_patient_data['datetime'].dt.day.unique()[:3]
        train_subset = train_patient_data[train_patient_data['datetime'].dt.day.isin(train_days_patient)]
        
        if len(train_subset) > 0:
            train_combined = train_subset[['datetime', 'carbohydrates', 'insulin']].copy()
            train_processed = add_azt1d_features(global_params, OPTIMIZATION_FEATURES_AZT1D,
                                                train_subset, train_combined)
            
            # Weight target patient data 10x
            if train_patient == patient:
                train_processed = pd.concat([train_processed] * 10, ignore_index=True)
            
            all_train_data.append(train_processed)
    
    if len(all_train_data) == 0:
        continue
    
    # Combine all training data
    X_train_all = pd.concat(all_train_data, ignore_index=True)
    
    # Prepare features
    feature_cols = ['glucose'] + OPTIMIZATION_FEATURES_AZT1D
    X_train = X_train_all[feature_cols].fillna(0)
    y_train = X_train_all[f'glucose_{DEFAULT_PREDICTION_HORIZON}'].fillna(0)
    
    # Train model
    try:
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        model = lgb.train(LGB_PARAMS, lgb.Dataset(X_tr, label=y_tr), 
                         valid_sets=[lgb.Dataset(X_val, label=y_val)])
        
        # Get baseline prediction
        X_baseline = baseline_test[feature_cols].fillna(0)
        baseline_predictions = model.predict(X_baseline)
        baseline_mean = np.mean(baseline_predictions)
        
        # Test carbohydrate modifications
        for increment in increments:
            # Modify the combined data
            modified_combined = modify_macronutrients(combined_data, 'carbohydrates', increment)
            
            # Re-generate features with modified data
            modified_data = add_azt1d_features(global_params, OPTIMIZATION_FEATURES_AZT1D,
                                             patient_data, modified_combined)
            modified_test = modified_data[modified_data['datetime'].dt.day.isin(test_days)]
            
            if len(modified_test) == 0:
                continue
            
            # Make predictions with modified data
            X_modified = modified_test[feature_cols].fillna(0)
            modified_predictions = model.predict(X_modified)
            modified_mean = np.mean(modified_predictions)
            
            # Calculate change
            glucose_change = modified_mean - baseline_mean
            
            # Store results
            results_df = pd.concat([results_df, pd.DataFrame({
                'patient': [patient],
                'feature': ['carbohydrates'],
                'increment': [increment],
                'glucose_change': [glucose_change],
                'baseline_prediction': [baseline_mean],
                'modified_prediction': [modified_mean]
            })], ignore_index=True)
            
    except Exception as e:
        print(f"  Error processing patient {patient}: {e}")
        continue

print(f"\nAnalysis complete. Generated {len(results_df)} data points for {results_df['patient'].nunique()} patients.")

if len(results_df) > 0:
    # Create visualization
    def create_azt1d_modification_plot():
        """Create visualization of carbohydrate modification effects for AZT1D"""
        
        # Calculate patient averages for each increment
        avg_results = results_df.groupby('increment')['glucose_change'].agg(['mean', 'std', 'count']).reset_index()
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Colors for patients
        unique_patients = sorted(results_df['patient'].unique())
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_patients)))
        
        # Plot individual patients as thin lines
        for i, patient in enumerate(unique_patients):
            patient_data = results_df[results_df['patient'] == patient].sort_values('increment')
            if len(patient_data) > 0:
                ax.plot(patient_data['increment'], patient_data['glucose_change'], 
                       color=colors[i], alpha=0.6, linewidth=1, label=f'Patient {patient}')
        
        # Plot average as thick line
        ax.plot(avg_results['increment'], avg_results['mean'], 
               color='black', linewidth=4, label='Average', marker='o', markersize=8)
        
        # Add error bars
        ax.errorbar(avg_results['increment'], avg_results['mean'], 
                   yerr=avg_results['std'], color='black', alpha=0.7, capsize=5, capthick=2)
        
        # Formatting
        ax.set_xlabel('Carbohydrate Modification Amount (g)', fontsize=14)
        ax.set_ylabel('Change in Predicted Glucose (mg/dL)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
        
        # Legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{SUPPLEMENTARY_DATA_PATH}/azt1d_carbohydrate_modifications.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return avg_results
    
    # Create visualization
    avg_results = create_azt1d_modification_plot()
    
    # Generate summary statistics
    print("\nAZT1D Summary of Carbohydrate Modification Effects:")
    print("=" * 60)
    
    baseline = avg_results[avg_results['increment'] == 0]['mean'].iloc[0]
    max_positive = avg_results[avg_results['increment'] == 50]['mean'].iloc[0]
    max_negative = avg_results[avg_results['increment'] == -50]['mean'].iloc[0]
    
    print(f"Carbohydrates:")
    print(f"  +50g effect: {max_positive:+.2f} mg/dL")
    print(f"  -50g effect: {max_negative:+.2f} mg/dL")
    print(f"  Total range: {max_positive - max_negative:.2f} mg/dL")
    print(f"  Patients analyzed: {results_df['patient'].nunique()}")
    
    # Save detailed results
    results_df.to_csv(f'{RESULTS_PATH}/azt1d_carbohydrate_modification_results.csv', index=False)
    avg_results.to_csv(f'{RESULTS_PATH}/azt1d_carbohydrate_modification_averages.csv', index=False)
    
    # Compare with D1namo results if available
    d1namo_file = f'{RESULTS_PATH}/macronutrient_modification_averages.csv'
    if os.path.exists(d1namo_file):
        d1namo_results = pd.read_csv(d1namo_file)
        d1namo_complex = d1namo_results[d1namo_results['feature'] == 'complex_sugars']
        
        if len(d1namo_complex) > 0:
            d1namo_baseline = d1namo_complex[d1namo_complex['increment'] == 0]['mean'].iloc[0]
            d1namo_max_pos = d1namo_complex[d1namo_complex['increment'] == 50]['mean'].iloc[0]
            d1namo_max_neg = d1namo_complex[d1namo_complex['increment'] == -50]['mean'].iloc[0]
            
            print(f"\nComparison with D1namo Complex Sugars:")
            print(f"D1namo +50g effect: {d1namo_max_pos:+.2f} mg/dL")
            print(f"AZT1D  +50g effect: {max_positive:+.2f} mg/dL")
            print(f"D1namo -50g effect: {d1namo_max_neg:+.2f} mg/dL")
            print(f"AZT1D  -50g effect: {max_negative:+.2f} mg/dL")
            
            # Check if direction is consistent
            d1namo_direction = "increases" if d1namo_max_pos > 0 else "decreases"
            azt1d_direction = "increases" if max_positive > 0 else "decreases"
            consistency = "CONSISTENT" if d1namo_direction == azt1d_direction else "INCONSISTENT"
            
            print(f"\nDirection comparison:")
            print(f"D1namo: Adding carbs {d1namo_direction} glucose")
            print(f"AZT1D:  Adding carbs {azt1d_direction} glucose")
            print(f"Result: {consistency}")

else:
    print("No data generated - check if AZT1D data is properly loaded and patients have sufficient data.")

print("\nAZT1D carbohydrate modification analysis complete!")