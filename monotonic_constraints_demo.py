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
LOAD_RESULTS = False  # Set to True to load existing results, False to recompute
increments = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]
prediction_horizon = 12  # 60 minutes

def get_azt1d_data(patient):
    """Load AZT1D data for a specific patient"""
    file_path = f"{AZT1D_DATA_PATH}/Subject {patient}/Subject {patient}.csv"
    df = pd.read_csv(file_path)
    df['patient'] = patient
    df['datetime'] = pd.to_datetime(df[AZT1D_COLUMNS['datetime']])
    df['glucose'] = df[AZT1D_COLUMNS['glucose']].fillna(0)
    df['carbohydrates'] = df[AZT1D_COLUMNS['carbohydrates']].fillna(0)
    df['insulin'] = df[AZT1D_COLUMNS['insulin']].fillna(0)
    df['correction'] = df[AZT1D_COLUMNS['correction']].fillna(0)
    df['hour'] = df['datetime'].dt.hour
    df['time'] = df['hour'] + df['datetime'].dt.minute / 60
    df = df[['patient', 'datetime', 'glucose', 'carbohydrates', 'insulin', 'correction', 'hour', 'time']].copy()
    df = df.sort_values('datetime').reset_index(drop=True)
    for horizon in PREDICTION_HORIZONS:
        df[f'glucose_{horizon}'] = df['glucose'].shift(-horizon) - df['glucose']
    df['glucose_change'] = df['glucose'] - df['glucose'].shift(1)
    df['glucose_change_projected'] = df['glucose_change'].rolling(6, min_periods=6).apply(lambda window: get_projected_value(window, 6))
    df['glucose_projected'] = df['glucose'].rolling(6, min_periods=6).apply(lambda window: get_projected_value(window, 6))
    df = df.dropna(subset=[f'glucose_24'])
    
    # Return glucose data and combined data (carbohydrates, insulin, correction)
    glucose_data = df.copy()
    combined_data = df[['datetime', 'carbohydrates', 'insulin', 'correction']].copy()
    return glucose_data, combined_data

def modify_macronutrients(combined_data, nutrient, amount, dataset='d1namo', time_window=None):
    """Modify a specific macronutrient or insulin by a given amount for all events."""
    modified_data = combined_data.copy()
    if time_window is not None:
        start_ts, end_ts = time_window
        in_window = (modified_data['datetime'] >= start_ts) & (modified_data['datetime'] <= end_ts)
    else:
        in_window = pd.Series(True, index=modified_data.index)
    
    if nutrient == 'insulin' or nutrient == 'correction':
        # Modify all insulin/correction events (where value > 0)
        mask = (modified_data[nutrient] > 0) & in_window
        if mask.any():
            modified_data.loc[mask, nutrient] += amount
            # Ensure values remain non-negative
            modified_data[nutrient] = modified_data[nutrient].clip(lower=0)
    elif dataset == 'd1namo':
        # Modify all food events (where any macronutrient is > 0)
        food_mask = ((modified_data['simple_sugars'] > 0) | (modified_data['complex_sugars'] > 0) | \
                    (modified_data['proteins'] > 0) | (modified_data['fats'] > 0) | (modified_data['dietary_fibers'] > 0)) & in_window
        
        if food_mask.any():
            modified_data.loc[food_mask, nutrient] += amount
            # Ensure values remain non-negative
            modified_data[nutrient] = modified_data[nutrient].clip(lower=0)
    else:  # AZT1D
        # Modify all carbohydrate events
        carb_mask = (modified_data['carbohydrates'] > 0) & in_window
        if carb_mask.any():
            modified_data.loc[carb_mask, nutrient] += amount
            modified_data[nutrient] = modified_data[nutrient].clip(lower=0)
    return modified_data

def run_modification_analysis_with_constraints(dataset='d1namo', use_monotonic=True):
    """Run macronutrient modification analysis for specified dataset with or without monotonic constraints"""
    
    # Set dataset-specific parameters
    if dataset == 'd1namo':
        patients = PATIENTS_D1NAMO
        meal_features = ['complex_sugars', 'insulin']  # Only complex sugars and insulin for D1NAMO
        optimization_features = OPTIMIZATION_FEATURES_D1NAMO
        features_to_remove = FEATURES_TO_REMOVE_D1NAMO
        load_data_func = get_d1namo_data
        params_file = f'results/bezier_params/d1namo_all_patient_bezier_params.json'
        training_days = 3
    else:  # azt1d
        patients = PATIENTS_AZT1D[:5]  # First 5 patients for AZT1D
        meal_features = ['carbohydrates', 'insulin']  # Carbohydrates and insulin for AZT1D
        optimization_features = OPTIMIZATION_FEATURES_AZT1D
        features_to_remove = FEATURES_TO_REMOVE_AZT1D
        load_data_func = get_azt1d_data
        params_file = f'results/bezier_params/azt1d_all_patient_bezier_params.json'
        training_days = 14  # 2 weeks for AZT1D
    
    # Load global Bézier parameters
    with open(params_file, 'r') as f:
        all_patient_params = json.load(f)
    
    # Calculate global parameters as average across all patients
    def calculate_global_params(all_patient_params, features):
        global_params = {}
        for feature in features:
            feature_params = []
            for patient_key in all_patient_params.keys():
                if feature in all_patient_params[patient_key]:
                    feature_params.append(all_patient_params[patient_key][feature])
            
            if feature_params:
                global_params[feature] = np.mean(feature_params, axis=0).tolist()
        return global_params
    
    global_params = calculate_global_params(all_patient_params, optimization_features)

    # Store results
    results_df = pd.DataFrame()
    
    # For each patient, analyze effect of macronutrient modifications
    for patient in patients:
        print(f"Processing {dataset} patient {patient}")
        
        # Get patient data
        glucose_data, combined_data = load_data_func(patient)
        
        # Get baseline prediction using original data
        baseline_data = add_temporal_features(global_params, optimization_features, glucose_data, combined_data, prediction_horizon)
        # Optional alternative horizon for D1namo insulin to visualize non-flat response
        PH_INS_VIS = 12  # 60 minutes for clearer insulin effect
        baseline_data_ins = add_temporal_features(global_params, optimization_features, glucose_data, combined_data, PH_INS_VIS)
        baseline_data_ins = None
        
        # Use only data from test days
        if dataset == 'd1namo':
            test_days = baseline_data['datetime'].dt.day.unique()[training_days:]
            baseline_test = baseline_data[baseline_data['datetime'].dt.day.isin(test_days)]
        else:  # azt1d
            all_days = sorted(baseline_data['datetime'].dt.day.unique())
            test_days = all_days[training_days:]  # Test on all days after first 14
            baseline_test = baseline_data[baseline_data['datetime'].dt.day.isin(test_days)]

        
        # Prepare training data (use data from all patients)
        all_train_data = []
        for train_patient in patients:
            train_glucose, train_combined = load_data_func(train_patient)
            train_processed = add_temporal_features(global_params, optimization_features, train_glucose, train_combined, prediction_horizon)
            
            # Use only training days
            if dataset == 'd1namo':
                train_days = train_processed['datetime'].dt.day.unique()[:training_days]
                train_subset = train_processed[train_processed['datetime'].dt.day.isin(train_days)]
            else:  # azt1d
                first_14_dates = sorted(train_processed['datetime'].dt.normalize().unique())[:training_days]
                train_subset = train_processed[train_processed['datetime'].dt.normalize().isin(first_14_dates)]
            all_train_data.append(train_subset)
        
        # Combine all training data
        X_train_all = pd.concat(all_train_data, ignore_index=True)
        
        # Remove features not used for prediction
        available_features = X_train_all.columns.difference(features_to_remove)
        X_train = X_train_all[available_features]
        y_train = X_train_all[f'glucose_{prediction_horizon}']
        
        # Train model with or without monotone constraints
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)
        
        # Create LGB parameters with or without monotone constraints
        lgb_params = LGB_PARAMS.copy()
        if use_monotonic:
            lgb_params['monotone_constraints'] = [MONOTONE_MAP.get(col, 0) for col in available_features]
        else:
            lgb_params['monotone_constraints'] = [0] * len(available_features)  # No constraints
        
        model = lgb.train(lgb_params, lgb.Dataset(X_tr, label=y_tr), 
                         valid_sets=[lgb.Dataset(X_val, label=y_val)])
        
        # Get baseline prediction
        X_baseline = baseline_test[available_features]
        baseline_predictions = model.predict(X_baseline)
        baseline_mean = np.mean(baseline_predictions)
        
        # For each macronutrient, test modifications
        for feature in meal_features:
            for increment in increments:
                # Modify the combined data (apply across all times to ensure visible effect)
                modified_combined = modify_macronutrients(combined_data, feature, increment, dataset, None)

                # Re-generate features with modified data
                if dataset == 'd1namo' and feature == 'insulin' and baseline_data_ins is not None:
                    # Use alternative PH for insulin visualization on D1namo
                    modified_data = add_temporal_features(global_params, optimization_features, glucose_data, modified_combined, PH_INS_VIS)
                    mod_test_ref = baseline_data_ins
                else:
                    modified_data = add_temporal_features(global_params, optimization_features, glucose_data, modified_combined, prediction_horizon)
                    mod_test_ref = baseline_data

                # Use only data from test days
                if dataset == 'd1namo':
                    ref_days = mod_test_ref['datetime'].dt.day.unique()[training_days:]
                    modified_test = modified_data[modified_data['datetime'].dt.day.isin(ref_days)]
                else:  # azt1d
                    all_days = sorted(mod_test_ref['datetime'].dt.day.unique())
                    ref_days = all_days[training_days:]  # Test on all days after first 14
                    modified_test = modified_data[modified_data['datetime'].dt.day.isin(ref_days)]

                
                # Make predictions with modified data
                X_modified = modified_test[available_features]
                modified_predictions = model.predict(X_modified)
                modified_mean = np.mean(modified_predictions)
                
                # Calculate change
                glucose_change = modified_mean - baseline_mean
                
                # Store results
                results_df = pd.concat([results_df, pd.DataFrame({
                    'dataset': [dataset],
                    'patient': [patient],
                    'feature': [feature],
                    'increment': [increment],
                    'glucose_change': [glucose_change],
                    'baseline_prediction': [baseline_mean],
                    'modified_prediction': [modified_mean],
                    'use_monotonic': [use_monotonic]
                })], ignore_index=True)
    
    print(f"{dataset.upper()} analysis complete. Generated {len(results_df)} data points.")
    return results_df
    
def create_combined_modification_plot(results_df):
    """Create a combined visualization showing both datasets with and without monotonic constraints"""
    
    # Create figure with 2 rows, 4 columns for the 8 subplots
    fig = plt.figure(figsize=(16, 8))
    
    # Colors for patients
    patient_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#DDA0DD', '#87CEEB']
    
    # Define features for each dataset
    d1namo_features = ['complex_sugars', 'insulin']
    azt1d_features = ['carbohydrates', 'insulin']
    
    # WITH MONOTONIC CONSTRAINTS FRAME (top row)
    with_constraints_results = results_df[results_df['use_monotonic'] == True]
    with_constraints_avg = with_constraints_results.groupby(['dataset', 'feature', 'increment'])['glucose_change'].mean().reset_index()
    
    # WITHOUT MONOTONIC CONSTRAINTS FRAME (bottom row)  
    without_constraints_results = results_df[results_df['use_monotonic'] == False]
    without_constraints_avg = without_constraints_results.groupby(['dataset', 'feature', 'increment'])['glucose_change'].mean().reset_index()
    
    # Create subplot grid (2x4) - top row for WITH constraints, bottom row for WITHOUT constraints
    axes = []
    for row in range(2):
        for col in range(4):
            ax = plt.subplot2grid((2, 4), (row, col), fig=fig)
            axes.append(ax)
    
    # Plot WITH constraints features (top row)
    patient_legend_elements = []
    
    # D1NAMO features (columns 0-1)
    for i, feature in enumerate(d1namo_features):
        ax = axes[i]
        dataset_results = with_constraints_results[with_constraints_results['dataset'] == 'd1namo']
        
        # Plot individual patients
        for j, patient in enumerate(PATIENTS_D1NAMO):
            patient_data = dataset_results[(dataset_results['feature'] == feature) & 
                                         (dataset_results['patient'] == patient)]
            if len(patient_data) > 0:
                patient_data = patient_data.sort_values('increment')
                line = ax.plot(
                    patient_data['increment'],
                    patient_data['glucose_change'],
                    color=patient_colors[j % len(patient_colors)],
                    alpha=0.45,
                    linewidth=1.2,
                    label=f'P{patient}',
                )
                # Collect legend elements only from first subplot
                if i == 0:
                    patient_legend_elements.append(
                        plt.Line2D([0], [0], color=patient_colors[j % len(patient_colors)], 
                                 alpha=0.45, linewidth=1.2, label=f'P{patient}')
                    )
        
        # Plot average
        feature_avg = with_constraints_avg[(with_constraints_avg['dataset'] == 'd1namo') & 
                                         (with_constraints_avg['feature'] == feature)].sort_values('increment')
        if len(feature_avg) > 0:
            ax.plot(
                feature_avg['increment'],
                feature_avg['glucose_change'],
                color='black', linewidth=2.5, label='Average', marker='o', markersize=3.5,
            )
        
        # Formatting with dataset in brackets
        feature_name = feature.replace("_", " ").title()
        ax.set_title(f'{feature_name} (D1NAMO, With Constraints)', fontsize=11, fontweight='bold')
        units = " (units)" if feature == "insulin" else " (g)"
        ax.set_xlabel(f'Amount{units}', fontsize=9)
        if i == 0:  # First column
            ax.set_ylabel('Δ Glucose (mg/dL)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.tick_params(axis='both', labelsize=8)
        
        # Add subplot label
        ax.text(0.02, 0.98, chr(65 + i), transform=ax.transAxes, fontsize=12, 
               fontweight='bold', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # AZT1D features (columns 2-3)
    for i, feature in enumerate(azt1d_features):
        ax = axes[i + 2]
        dataset_results = with_constraints_results[with_constraints_results['dataset'] == 'azt1d']
        
        # Plot individual patients
        for j, patient in enumerate(PATIENTS_AZT1D[:5]):
            patient_data = dataset_results[(dataset_results['feature'] == feature) & 
                                         (dataset_results['patient'] == patient)]
            if len(patient_data) > 0:
                patient_data = patient_data.sort_values('increment')
                ax.plot(
                    patient_data['increment'],
                    patient_data['glucose_change'],
                    color=patient_colors[j % len(patient_colors)],
                    alpha=0.45,
                    linewidth=1.2,
                    label=f'P{patient}',
                )
        
        # Plot average
        feature_avg = with_constraints_avg[(with_constraints_avg['dataset'] == 'azt1d') & 
                                         (with_constraints_avg['feature'] == feature)].sort_values('increment')
        if len(feature_avg) > 0:
            ax.plot(
                feature_avg['increment'],
                feature_avg['glucose_change'],
                color='black', linewidth=2.5, label='Average', marker='o', markersize=3.5,
            )
        
        # Formatting with dataset in brackets
        feature_name = feature.replace("_", " ").title()
        ax.set_title(f'{feature_name} (AZT1D, With Constraints)', fontsize=11, fontweight='bold')
        units = " (units)" if feature == "insulin" else " (g)"
        ax.set_xlabel(f'Amount{units}', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.tick_params(axis='both', labelsize=8)
        
        # Add subplot label
        ax.text(0.02, 0.98, chr(67 + i), transform=ax.transAxes, fontsize=12, 
               fontweight='bold', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot WITHOUT constraints features (bottom row)
    # D1NAMO features (columns 0-1)
    for i, feature in enumerate(d1namo_features):
        ax = axes[i + 4]
        dataset_results = without_constraints_results[without_constraints_results['dataset'] == 'd1namo']
        
        # Plot individual patients
        for j, patient in enumerate(PATIENTS_D1NAMO):
            patient_data = dataset_results[(dataset_results['feature'] == feature) & 
                                         (dataset_results['patient'] == patient)]
            if len(patient_data) > 0:
                patient_data = patient_data.sort_values('increment')
                ax.plot(
                    patient_data['increment'],
                    patient_data['glucose_change'],
                    color=patient_colors[j % len(patient_colors)],
                    alpha=0.45,
                    linewidth=1.2,
                    label=f'P{patient}',
                )
        
        # Plot average
        feature_avg = without_constraints_avg[(without_constraints_avg['dataset'] == 'd1namo') & 
                                            (without_constraints_avg['feature'] == feature)].sort_values('increment')
        if len(feature_avg) > 0:
            ax.plot(
                feature_avg['increment'],
                feature_avg['glucose_change'],
                color='black', linewidth=2.5, label='Average', marker='o', markersize=3.5,
            )
        
        # Formatting with dataset in brackets
        feature_name = feature.replace("_", " ").title()
        ax.set_title(f'{feature_name} (D1NAMO, Without Constraints)', fontsize=11, fontweight='bold')
        units = " (units)" if feature == "insulin" else " (g)"
        ax.set_xlabel(f'Amount{units}', fontsize=9)
        if i == 0:  # First column
            ax.set_ylabel('Δ Glucose (mg/dL)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.tick_params(axis='both', labelsize=8)
        
        # Add subplot label
        ax.text(0.02, 0.98, chr(69 + i), transform=ax.transAxes, fontsize=12, 
               fontweight='bold', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # AZT1D features (columns 2-3)
    for i, feature in enumerate(azt1d_features):
        ax = axes[i + 6]
        dataset_results = without_constraints_results[without_constraints_results['dataset'] == 'azt1d']
        
        # Plot individual patients
        for j, patient in enumerate(PATIENTS_AZT1D[:5]):
            patient_data = dataset_results[(dataset_results['feature'] == feature) & 
                                         (dataset_results['patient'] == patient)]
            if len(patient_data) > 0:
                patient_data = patient_data.sort_values('increment')
                ax.plot(
                    patient_data['increment'],
                    patient_data['glucose_change'],
                    color=patient_colors[j % len(patient_colors)],
                    alpha=0.45,
                    linewidth=1.2,
                    label=f'P{patient}',
                )
        
        # Plot average
        feature_avg = without_constraints_avg[(without_constraints_avg['dataset'] == 'azt1d') & 
                                            (without_constraints_avg['feature'] == feature)].sort_values('increment')
        if len(feature_avg) > 0:
            ax.plot(
                feature_avg['increment'],
                feature_avg['glucose_change'],
                color='black', linewidth=2.5, label='Average', marker='o', markersize=3.5,
            )
        
        # Formatting with dataset in brackets
        feature_name = feature.replace("_", " ").title()
        ax.set_title(f'{feature_name} (AZT1D, Without Constraints)', fontsize=11, fontweight='bold')
        units = " (units)" if feature == "insulin" else " (g)"
        ax.set_xlabel(f'Amount{units}', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.tick_params(axis='both', labelsize=8)
        
        # Add subplot label
        ax.text(0.02, 0.98, chr(71 + i), transform=ax.transAxes, fontsize=12, 
               fontweight='bold', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    # Create shared legend with individual patient IDs
    legend_elements = patient_legend_elements + [
        plt.Line2D([0], [0], color='black', linewidth=2.5, marker='o', markersize=3.5, label='Average'),
    ]
    fig.legend(legend_elements, [e.get_label() for e in legend_elements],
               loc='lower center', ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    # Save figure
    plt.savefig('manuscript/images/supplementary_data/monotonic_constraints_comparison.eps', dpi=300, bbox_inches='tight')
    plt.close()
    
    return with_constraints_avg, without_constraints_avg

# Main execution
if __name__ == "__main__":
    # Run analysis for both datasets and constraint types
    all_results = []
    
    # D1NAMO dataset
    print("Running D1NAMO analysis WITH monotonic constraints...")
    d1namo_with_constraints = run_modification_analysis_with_constraints('d1namo', use_monotonic=True)
    all_results.append(d1namo_with_constraints)
    
    print("\nRunning D1NAMO analysis WITHOUT monotonic constraints...")
    d1namo_without_constraints = run_modification_analysis_with_constraints('d1namo', use_monotonic=False)
    all_results.append(d1namo_without_constraints)
    
    # AZT1D dataset
    print("\nRunning AZT1D analysis WITH monotonic constraints...")
    azt1d_with_constraints = run_modification_analysis_with_constraints('azt1d', use_monotonic=True)
    all_results.append(azt1d_with_constraints)
    
    print("\nRunning AZT1D analysis WITHOUT monotonic constraints...")
    azt1d_without_constraints = run_modification_analysis_with_constraints('azt1d', use_monotonic=False)
    all_results.append(azt1d_without_constraints)
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Create combined visualization
    print("\nCreating combined visualization...")
    with_avg, without_avg = create_combined_modification_plot(combined_results)
    
    # Generate summary statistics
    print("\n" + "="*60)
    print("MONOTONIC CONSTRAINTS COMPARISON SUMMARY")
    print("="*60)
    
    # Compare results for each dataset and feature
    datasets_features = {
        'd1namo': ['complex_sugars', 'insulin'],
        'azt1d': ['carbohydrates', 'insulin']
    }
    
    for dataset, features in datasets_features.items():
        print(f"\n{dataset.upper()} DATASET:")
        for feature in features:
            print(f"\n{feature.replace('_', ' ').title()}:")
            
            # Get data for both constraint types
            with_feature = with_avg[(with_avg['dataset'] == dataset) & (with_avg['feature'] == feature)]
            without_feature = without_avg[(without_avg['dataset'] == dataset) & (without_avg['feature'] == feature)]
            
            if len(with_feature) > 0 and len(without_feature) > 0:
                # Compare +50 increment effects
                with_50 = with_feature[with_feature['increment'] == 50]['glucose_change'].iloc[0]
                without_50 = without_feature[without_feature['increment'] == 50]['glucose_change'].iloc[0]
                
                # Compare -50 increment effects
                with_neg50 = with_feature[with_feature['increment'] == -50]['glucose_change'].iloc[0]
                without_neg50 = without_feature[without_feature['increment'] == -50]['glucose_change'].iloc[0]
                
                print(f"  +50 effect: With constraints {with_50:+.2f} mg/dL, Without constraints {without_50:+.2f} mg/dL")
                print(f"  -50 effect: With constraints {with_neg50:+.2f} mg/dL, Without constraints {without_neg50:+.2f} mg/dL")
                
                # Check monotonicity
                with_monotonic = all(with_feature.sort_values('increment')['glucose_change'].diff().dropna() >= 0) if feature != 'insulin' else all(with_feature.sort_values('increment')['glucose_change'].diff().dropna() <= 0)
                without_monotonic = all(without_feature.sort_values('increment')['glucose_change'].diff().dropna() >= 0) if feature != 'insulin' else all(without_feature.sort_values('increment')['glucose_change'].diff().dropna() <= 0)
                
                print(f"  Monotonicity: With constraints {'✓' if with_monotonic else '✗'}, Without constraints {'✓' if without_monotonic else '✗'}")
    
    # Save detailed results
    combined_results.to_csv('results/monotonic_constraints_comparison_results.csv', index=False)
    print(f"\nResults saved to results/monotonic_constraints_comparison_results.csv")
    print("\nAnalysis complete. Visualization saved to manuscript/images/supplementary_data/monotonic_constraints_comparison.eps")
