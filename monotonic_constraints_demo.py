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
        meal_features = ['simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers', 'insulin']
        optimization_features = OPTIMIZATION_FEATURES_D1NAMO
        features_to_remove = FEATURES_TO_REMOVE_D1NAMO
        load_data_func = get_d1namo_data
        params_file = f'results/bezier_params/d1namo_all_patient_bezier_params.json'
        training_days = 3
    else:  # azt1d
        patients = PATIENTS_AZT1D[:3]  # Use first 3 AZT1D subjects for faster dev
        meal_features = ['carbohydrates', 'insulin', 'correction']
        optimization_features = OPTIMIZATION_FEATURES_AZT1D
        features_to_remove = FEATURES_TO_REMOVE_AZT1D
        params_file = f'results/bezier_params/azt1d_all_patient_bezier_params.json'
        training_days = 14
        
        def load_azt1d_data(patient):
            """Load AZT1D data for a specific patient."""
            file_path = f"{AZT1D_DATA_PATH}/Subject {patient}/Subject {patient}.csv"
            df = pd.read_csv(file_path)
            
            # Keep only the columns we need
            df['datetime'] = pd.to_datetime(df[AZT1D_COLUMNS['datetime']])
            df['glucose'] = df[AZT1D_COLUMNS['glucose']].fillna(0)
            df['carbohydrates'] = df[AZT1D_COLUMNS['carbohydrates']].fillna(0)
            df['insulin'] = df[AZT1D_COLUMNS['insulin']].fillna(0)
            df['correction'] = df[AZT1D_COLUMNS['correction']].fillna(0)
            df['hour'] = df['datetime'].dt.hour
            df['time'] = df['hour'] + df['datetime'].dt.minute / 60
            
            # Keep only needed columns
            df = df[['datetime', 'glucose', 'carbohydrates', 'insulin', 'correction', 'hour', 'time']].copy()
            
            # Add prediction horizon features
            for horizon in PREDICTION_HORIZONS:
                df[f'glucose_{horizon}'] = df['glucose'].shift(-horizon) - df['glucose']
            
            # Add glucose change and projected features
            df['glucose_change'] = df['glucose'] - df['glucose'].shift(1)
            df['glucose_change_projected'] = df['glucose_change'].rolling(6, min_periods=6).apply(lambda window: get_projected_value(window, 6))
            df['glucose_projected'] = df['glucose'].rolling(6, min_periods=6).apply(lambda window: get_projected_value(window, 6))
            df = df.dropna(subset=[f'glucose_24'])
            
            # Create combined data for temporal features
            combined_data = df[['datetime', 'carbohydrates', 'insulin', 'correction']].copy()
            
            return df, combined_data
        
        load_data_func = load_azt1d_data
    
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
        baseline_data_ins = None
        PH_INS_VIS = 12  # 60 minutes for clearer insulin effect
        if dataset == 'd1namo':
            baseline_data_ins = add_temporal_features(global_params, optimization_features, glucose_data, combined_data, PH_INS_VIS)
        
        # Use only data from test days
        if dataset == 'd1namo':
            test_days = baseline_data['datetime'].dt.day.unique()[training_days:]
            baseline_test = baseline_data[baseline_data['datetime'].dt.day.isin(test_days)]
        else:  # AZT1D
            test_dates = sorted(baseline_data['datetime'].dt.normalize().unique())[training_days:training_days+7]  # Use week after training
            baseline_test = baseline_data[baseline_data['datetime'].dt.normalize().isin(test_dates)]
        
        if len(baseline_test) == 0:
            continue
            
        # Prepare training data (use data from all patients)
        all_train_data = []
        for train_patient in patients:
            train_glucose, train_combined = load_data_func(train_patient)
            train_processed = add_temporal_features(global_params, optimization_features, train_glucose, train_combined, prediction_horizon)
            
            # Use only training days
            if dataset == 'd1namo':
                train_days = train_processed['datetime'].dt.day.unique()[:training_days]
                train_subset = train_processed[train_processed['datetime'].dt.day.isin(train_days)]
            else:  # AZT1D
                train_dates = sorted(train_processed['datetime'].dt.normalize().unique())[:training_days]
                train_subset = train_processed[train_processed['datetime'].dt.normalize().isin(train_dates)]
            
            if len(train_subset) > 0:
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

                if dataset == 'd1namo':
                    ref_days = mod_test_ref['datetime'].dt.day.unique()[training_days:]
                    modified_test = modified_data[modified_data['datetime'].dt.day.isin(ref_days)]
                else:  # AZT1D
                    ref_dates = sorted(mod_test_ref['datetime'].dt.normalize().unique())[training_days:training_days+7]
                    modified_test = modified_data[modified_data['datetime'].dt.normalize().isin(ref_dates)]
                
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
    """Create a combined visualization showing both with and without monotonic constraints"""
    
    # Create figure with 2 rows, 3 columns for the 3 key features
    fig = plt.figure(figsize=(12, 8))
    
    # Colors for patients
    patient_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#DDA0DD', '#87CEEB']
    
    # Focus on the 3 key features that best demonstrate monotonic constraints
    key_features = ['simple_sugars', 'complex_sugars', 'insulin']
    
    # WITH MONOTONIC CONSTRAINTS FRAME (top row)
    with_constraints_results = results_df[results_df['use_monotonic'] == True]
    with_constraints_avg = with_constraints_results.groupby(['feature', 'increment'])['glucose_change'].mean().reset_index()
    d1namo_patients = PATIENTS_D1NAMO
    
    # Create subplot grid (2x3) - top row for WITH constraints
    with_axes = []
    for i in range(3):
        ax = plt.subplot2grid((2, 3), (0, i), fig=fig)
        with_axes.append(ax)
    
    # WITHOUT MONOTONIC CONSTRAINTS FRAME (bottom row)  
    without_constraints_results = results_df[results_df['use_monotonic'] == False]
    without_constraints_avg = without_constraints_results.groupby(['feature', 'increment'])['glucose_change'].mean().reset_index()
    
    # Create subplot grid (2x3) - bottom row for WITHOUT constraints
    without_axes = []
    for i in range(3):
        ax = plt.subplot2grid((2, 3), (1, i), fig=fig)
        without_axes.append(ax)
    
    # Plot WITH constraints features (top row)
    with_individual_label_added = False
    for i, feature in enumerate(key_features):
        ax = with_axes[i]
        
        # Plot individual patients
        for j, patient in enumerate(d1namo_patients):
            patient_data = with_constraints_results[(with_constraints_results['feature'] == feature) & 
                                         (with_constraints_results['patient'] == patient)]
            if len(patient_data) > 0:
                patient_data = patient_data.sort_values('increment')
                ax.plot(
                    patient_data['increment'],
                    patient_data['glucose_change'],
                    color=patient_colors[j % len(patient_colors)],
                    alpha=0.45,
                    linewidth=1.2,
                    label=('Individuals' if not with_individual_label_added else None),
                )
                with_individual_label_added = True
        
        # Plot average
        feature_avg = with_constraints_avg[with_constraints_avg['feature'] == feature].sort_values('increment')
        if len(feature_avg) > 0:
            ax.plot(
                feature_avg['increment'],
                feature_avg['glucose_change'],
                color='black', linewidth=2.5, label='Average', marker='o', markersize=3.5,
            )
        
        # Formatting with constraint type in brackets
        ax.set_title(f'{feature.replace("_", " ").title()} (With Constraints)', fontsize=11, fontweight='bold')
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
    
    # Plot WITHOUT constraints features (bottom row)
    without_individual_label_added = False
    for i, feature in enumerate(key_features):
        ax = without_axes[i]
        
        # Plot individual patients
        for j, patient in enumerate(d1namo_patients):
            patient_data = without_constraints_results[(without_constraints_results['feature'] == feature) & 
                                         (without_constraints_results['patient'] == patient)]
            if len(patient_data) > 0:
                patient_data = patient_data.sort_values('increment')
                ax.plot(
                    patient_data['increment'],
                    patient_data['glucose_change'],
                    color=patient_colors[j % len(patient_colors)],
                    alpha=0.45,
                    linewidth=1.2,
                    label=('Individuals' if not without_individual_label_added else None),
                )
                without_individual_label_added = True
        
        # Plot average
        feature_avg = without_constraints_avg[without_constraints_avg['feature'] == feature].sort_values('increment')
        if len(feature_avg) > 0:
            ax.plot(
                feature_avg['increment'],
                feature_avg['glucose_change'],
                color='black', linewidth=2.5, label='Average', marker='o', markersize=3.5,
            )
        
        # Formatting with constraint type in brackets
        ax.set_title(f'{feature.replace("_", " ").title()} (Without Constraints)', fontsize=11, fontweight='bold')
        units = " (units)" if feature == "insulin" else " (g)"
        ax.set_xlabel(f'Amount{units}', fontsize=9)
        if i == 0:  # First column
            ax.set_ylabel('Δ Glucose (mg/dL)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.tick_params(axis='both', labelsize=8)
        
        # Add subplot label
        ax.text(0.02, 0.98, chr(68 + i), transform=ax.transAxes, fontsize=12, 
               fontweight='bold', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    # Create shared legend
    legend_elements = [
        plt.Line2D([0], [0], color=patient_colors[0], alpha=0.45, linewidth=1.2, label='Individuals'),
        plt.Line2D([0], [0], color='black', linewidth=2.5, marker='o', markersize=3.5, label='Average'),
    ]
    fig.legend(legend_elements, [e.get_label() for e in legend_elements],
               loc='lower center', ncol=2, fontsize=12,
               bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    
    # Save figure
    plt.savefig('manuscript/images/results/monotonic_constraints_comparison.eps', dpi=300, bbox_inches='tight')
    plt.close()
    
    return with_constraints_avg, without_constraints_avg

# Main execution
if __name__ == "__main__":
    # Run analysis for both constraint types
    print("Running D1namo analysis WITH monotonic constraints...")
    with_constraints_results = run_modification_analysis_with_constraints('d1namo', use_monotonic=True)
    
    print("\nRunning D1namo analysis WITHOUT monotonic constraints...")
    without_constraints_results = run_modification_analysis_with_constraints('d1namo', use_monotonic=False)
    
    # Combine results
    all_results = pd.concat([with_constraints_results, without_constraints_results], ignore_index=True)
    
    # Create combined visualization
    print("\nCreating combined visualization...")
    with_avg, without_avg = create_combined_modification_plot(all_results)
    
    # Generate summary statistics
    print("\n" + "="*60)
    print("MONOTONIC CONSTRAINTS COMPARISON SUMMARY")
    print("="*60)
    
    # Compare results for each feature
    d1namo_features = ['simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers', 'insulin']
    
    for feature in d1namo_features:
        print(f"\n{feature.replace('_', ' ').title()}:")
        
        # Get data for both constraint types
        with_feature = with_avg[with_avg['feature'] == feature]
        without_feature = without_avg[without_avg['feature'] == feature]
        
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
    all_results.to_csv('results/monotonic_constraints_comparison_results.csv', index=False)
    print(f"\nResults saved to results/monotonic_constraints_comparison_results.csv")
    print("\nAnalysis complete. Visualization saved to manuscript/images/results/monotonic_constraints_comparison.eps")
