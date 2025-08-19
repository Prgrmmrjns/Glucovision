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

def run_modification_analysis(dataset='d1namo'):
    """Run macronutrient modification analysis for specified dataset."""
    
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
        patients = PATIENTS_AZT1D  # Use all AZT1D subjects
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
        
        # Train model with monotone constraints
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)
        
        # Create LGB parameters with monotone constraints
        lgb_params_with_constraints = LGB_PARAMS.copy()
        lgb_params_with_constraints['monotone_constraints'] = [MONOTONE_MAP.get(col, 0) for col in available_features]
        
        model = lgb.train(lgb_params_with_constraints, lgb.Dataset(X_tr, label=y_tr), 
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
                    'modified_prediction': [modified_mean]
                })], ignore_index=True)
    
    print(f"{dataset.upper()} analysis complete. Generated {len(results_df)} data points.")
    return results_df

def create_combined_modification_plot(results_df):
    """Create a combined visualization showing both D1namo and AZT1D macronutrient modification effects"""
    
    # Create figure with separate frames for each dataset using a 3-row grid
    fig = plt.figure(figsize=(15, 11))
    
    # Colors for patients
    patient_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#DDA0DD', '#87CEEB']
    
    # D1NAMO DATASET FRAME (top 2/3 of figure)
    d1namo_results = results_df[results_df['dataset'] == 'd1namo']
    d1namo_avg = d1namo_results.groupby(['feature', 'increment'])['glucose_change'].mean().reset_index()
    d1namo_features = ['simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers', 'insulin']
    d1namo_patients = PATIENTS_D1NAMO
    
    # Create D1namo subplot grid (2x3) in upper 2 rows of a 3x3 grid
    d1_axes = []
    for i in range(6):
        row = i // 3
        col = i % 3
        ax = plt.subplot2grid((3, 3), (row, col), fig=fig)
        d1_axes.append(ax)
    
    # AZT1D DATASET FRAME (bottom 1/3 of figure)  
    azt1d_results = results_df[results_df['dataset'] == 'azt1d']
    azt1d_avg = azt1d_results.groupby(['feature', 'increment'])['glucose_change'].mean().reset_index()
    azt1d_features = ['carbohydrates', 'insulin', 'correction']
    azt1d_patients = PATIENTS_AZT1D
    
    # Create AZT1D subplot grid (1x3) on third row of 3x3 grid
    azt_axes = []
    for i in range(3):
        ax = plt.subplot2grid((3, 3), (2, i), fig=fig)
        azt_axes.append(ax)
    
    # Plot D1namo features
    d1_individual_label_added = False
    for i, feature in enumerate(d1namo_features):
        ax = d1_axes[i]
        
        # Plot individual patients
        for j, patient in enumerate(d1namo_patients):
            patient_data = d1namo_results[(d1namo_results['feature'] == feature) & 
                                         (d1namo_results['patient'] == patient)]
            if len(patient_data) > 0:
                patient_data = patient_data.sort_values('increment')
                ax.plot(
                    patient_data['increment'],
                    patient_data['glucose_change'],
                    color=patient_colors[j % len(patient_colors)],
                    alpha=0.45,
                    linewidth=1.2,
                    label=('Individuals' if not d1_individual_label_added else None),
                )
                d1_individual_label_added = True
        
        # Plot average
        feature_avg = d1namo_avg[d1namo_avg['feature'] == feature].sort_values('increment')
        if len(feature_avg) > 0:
            ax.plot(
                feature_avg['increment'],
                feature_avg['glucose_change'],
                color='black', linewidth=2.5, label='Average', marker='o', markersize=3.5,
            )
        
        # Formatting with dataset name in brackets
        ax.set_title(f'{feature.replace("_", " ").title()} (D1namo)', fontsize=11, fontweight='bold')
        units = " (units)" if feature == "insulin" else " (g)"
        ax.set_xlabel(f'Amount{units}', fontsize=9)
        if i % 3 == 0:  # First column
            ax.set_ylabel('Δ Glucose (mg/dL)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.tick_params(axis='both', labelsize=8)
        
        # Add subplot label
        ax.text(0.02, 0.98, chr(65 + i), transform=ax.transAxes, fontsize=12, 
               fontweight='bold', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # No per-subplot dataset caption (use figure-level caption)
    
    # Plot AZT1D features
    azt_individual_label_added = False
    for i, feature in enumerate(azt1d_features):
        ax = azt_axes[i]
        
        # Plot individual patients
        for j, patient in enumerate(azt1d_patients):
            patient_data = azt1d_results[(azt1d_results['feature'] == feature) & 
                                        (azt1d_results['patient'] == patient)]
            if len(patient_data) > 0:
                patient_data = patient_data.sort_values('increment')
                ax.plot(
                    patient_data['increment'],
                    patient_data['glucose_change'],
                    color=patient_colors[j % len(patient_colors)],
                    alpha=0.45,
                    linewidth=1.2,
                    label=('Individuals' if not azt_individual_label_added else None),
                )
                azt_individual_label_added = True
        
        # Plot average
        feature_avg = azt1d_avg[azt1d_avg['feature'] == feature].sort_values('increment')
        if len(feature_avg) > 0:
            ax.plot(
                feature_avg['increment'],
                feature_avg['glucose_change'],
                color='black', linewidth=2.5, label='Average', marker='o', markersize=3.5,
            )
        
        # Formatting with dataset name in brackets
        ax.set_title(f'{feature.replace("_", " ").title()} (AZT1D)', fontsize=11, fontweight='bold')
        units = " (units)" if feature in ["insulin", "correction"] else " (g)"
        ax.set_xlabel(f'Amount{units}', fontsize=9)
        if i == 0:
            ax.set_ylabel('Δ Glucose (mg/dL)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.tick_params(axis='both', labelsize=8)
        
        # Add subplot label
        ax.text(0.02, 0.98, chr(71 + i), transform=ax.transAxes, fontsize=12, 
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
    plt.subplots_adjust(bottom=0.1)
    
    # Save figure
    plt.savefig('manuscript/images/results/combined_macronutrient_modifications.eps', dpi=300, bbox_inches='tight')
    plt.close()
    
    return d1namo_avg, azt1d_avg

# Main execution
if __name__ == "__main__":
    # Run analysis for both datasets
    print("Running D1namo analysis...")
    d1namo_results = run_modification_analysis('d1namo')
    
    print("\nRunning AZT1D analysis...")
    azt1d_results = run_modification_analysis('azt1d')
    
    # Combine results
    all_results = pd.concat([d1namo_results, azt1d_results], ignore_index=True)
    
    # Create combined visualization
    print("\nCreating combined visualization...")
    d1namo_avg, azt1d_avg = create_combined_modification_plot(all_results)
    
    # Generate summary statistics for both datasets
    for dataset in ['d1namo', 'azt1d']:
        print(f"\n{dataset.upper()} - Summary of Macronutrient Modification Effects:")
        print("=" * 60)
        
        dataset_results = all_results[all_results['dataset'] == dataset]
        dataset_avg = dataset_results.groupby(['feature', 'increment'])['glucose_change'].mean().reset_index()
        
        if dataset == 'd1namo':
            features = ['simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers', 'insulin']
        else:
            features = ['carbohydrates', 'insulin']
        
        for feature in features:
            feature_data = dataset_avg[dataset_avg['feature'] == feature]
            if len(feature_data) > 0:
                baseline = feature_data[feature_data['increment'] == 0]['glucose_change'].iloc[0] if len(feature_data[feature_data['increment'] == 0]) > 0 else 0
                max_positive = feature_data[feature_data['increment'] == 50]['glucose_change'].iloc[0] if len(feature_data[feature_data['increment'] == 50]) > 0 else 0
                max_negative = feature_data[feature_data['increment'] == -50]['glucose_change'].iloc[0] if len(feature_data[feature_data['increment'] == -50]) > 0 else 0
                
                units = "units" if feature in ["insulin", "correction"] else "g"
                print(f"\n{feature.replace('_', ' ').title()}:")
                print(f"  +50{units} effect: {max_positive:+.2f} mg/dL")
                print(f"  -50{units} effect: {max_negative:+.2f} mg/dL")
                print(f"  Total range: {max_positive - max_negative:.2f} mg/dL")
    
    # Save detailed results
    all_results.to_csv('results/macronutrient_modification_results.csv', index=False)
    print(f"\nResults saved to results/macronutrient_modification_results.csv")

    # ==========================
    # Macronutrient Modification Insights
    # ==========================
    
    def analyze_modification_insights(all_results):
        """Analyze macronutrient modification patterns and generate insights"""
        
        insights = []
        
        # 1. Overall effect magnitudes
        for dataset in ['d1namo', 'azt1d']:
            dataset_results = all_results[all_results['dataset'] == dataset]
            dataset_avg = dataset_results.groupby(['feature', 'increment'])['glucose_change'].mean().reset_index()
            
            if dataset == 'd1namo':
                features = ['simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers', 'insulin']
            else:
                features = ['carbohydrates', 'insulin', 'correction']
            
            for feature in features:
                feature_data = dataset_avg[dataset_avg['feature'] == feature]
                if len(feature_data) > 0:
                    max_positive = feature_data[feature_data['increment'] == 50]['glucose_change'].iloc[0] if len(feature_data[feature_data['increment'] == 50]) > 0 else 0
                    max_negative = feature_data[feature_data['increment'] == -50]['glucose_change'].iloc[0] if len(feature_data[feature_data['increment'] == -50]) > 0 else 0
                    total_range = max_positive - max_negative
                    
                    insights.append(f"{dataset.upper()} {feature}: +50 effect {max_positive:+.2f} mg/dL, -50 effect {max_negative:+.2f} mg/dL, range {total_range:.2f} mg/dL")
        
        # 2. Patient variability analysis
        for dataset in ['d1namo', 'azt1d']:
            dataset_results = all_results[all_results['dataset'] == dataset]
            
            if dataset == 'd1namo':
                features = ['simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers', 'insulin']
            else:
                features = ['carbohydrates', 'insulin', 'correction']
            
            for feature in features:
                feature_data = dataset_results[dataset_results['feature'] == feature]
                if len(feature_data) > 0:
                    # Calculate coefficient of variation for +50 increment
                    pos_50_data = feature_data[feature_data['increment'] == 50]['glucose_change']
                    if len(pos_50_data) > 1:
                        cv = pos_50_data.std() / abs(pos_50_data.mean()) * 100 if pos_50_data.mean() != 0 else 0
                        insights.append(f"{dataset.upper()} {feature} +50 CV: {cv:.1f}%")
        
        # 3. Cross-dataset comparison
        d1namo_avg = all_results[all_results['dataset'] == 'd1namo'].groupby(['feature', 'increment'])['glucose_change'].mean().reset_index()
        azt1d_avg = all_results[all_results['dataset'] == 'azt1d'].groupby(['feature', 'increment'])['glucose_change'].mean().reset_index()
        
        # Compare insulin effects
        d1_insulin = d1namo_avg[d1namo_avg['feature'] == 'insulin']
        a_insulin = azt1d_avg[azt1d_avg['feature'] == 'insulin']
        
        if len(d1_insulin) > 0 and len(a_insulin) > 0:
            d1_ins_50 = d1_insulin[d1_insulin['increment'] == 50]['glucose_change'].iloc[0]
            a_ins_50 = a_insulin[a_insulin['increment'] == 50]['glucose_change'].iloc[0]
            insights.append(f"Insulin +50: D1namo {d1_ins_50:+.2f} mg/dL vs AZT1D {a_ins_50:+.2f} mg/dL")
        
        # 4. Most impactful features
        for dataset in ['d1namo', 'azt1d']:
            dataset_avg = all_results[all_results['dataset'] == dataset].groupby(['feature', 'increment'])['glucose_change'].mean().reset_index()
            
            if dataset == 'd1namo':
                features = ['simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers', 'insulin']
            else:
                features = ['carbohydrates', 'insulin', 'correction']
            
            max_range = 0
            max_feature = None
            for feature in features:
                feature_data = dataset_avg[dataset_avg['feature'] == feature]
                if len(feature_data) > 0:
                    max_positive = feature_data[feature_data['increment'] == 50]['glucose_change'].iloc[0] if len(feature_data[feature_data['increment'] == 50]) > 0 else 0
                    max_negative = feature_data[feature_data['increment'] == -50]['glucose_change'].iloc[0] if len(feature_data[feature_data['increment'] == -50]) > 0 else 0
                    total_range = abs(max_positive - max_negative)
                    
                    if total_range > max_range:
                        max_range = total_range
                        max_feature = feature
            
            if max_feature:
                insights.append(f"{dataset.upper()} most impactful: {max_feature} (range {max_range:.2f} mg/dL)")
        
        return insights
    
    def generate_latex_insights(all_results):
        """Generate LaTeX-ready insights paragraph"""
        
        latex_lines = []
        
        # D1namo insights
        d1namo_results = all_results[all_results['dataset'] == 'd1namo']
        d1namo_avg = d1namo_results.groupby(['feature', 'increment'])['glucose_change'].mean().reset_index()
        
        # Find most impactful features for D1namo
        d1namo_features = ['simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers', 'insulin']
        d1namo_effects = {}
        
        for feature in d1namo_features:
            feature_data = d1namo_avg[d1namo_avg['feature'] == feature]
            if len(feature_data) > 0:
                max_positive = feature_data[feature_data['increment'] == 50]['glucose_change'].iloc[0] if len(feature_data[feature_data['increment'] == 50]) > 0 else 0
                max_negative = feature_data[feature_data['increment'] == -50]['glucose_change'].iloc[0] if len(feature_data[feature_data['increment'] == -50]) > 0 else 0
                d1namo_effects[feature] = (max_positive, max_negative)
        
        # Sort by total range
        d1namo_sorted = sorted(d1namo_effects.items(), key=lambda x: abs(x[1][0] - x[1][1]), reverse=True)
        
        if len(d1namo_sorted) >= 3:
            top1, top2, top3 = d1namo_sorted[:3]
            latex_lines.append(f"For D1namo at 60 minutes, {top1[0].replace('_', ' ')} produced the largest glucose modifications ({top1[1][0]:+.2f} mg/dL at +50 g; {top1[1][1]:+.2f} mg/dL at -50 g, total range {abs(top1[1][0] - top1[1][1]):.2f} mg/dL).")
            latex_lines.append(f"{top2[0].replace('_', ' ').title()} resulted in {top2[1][0]:+.2f} mg/dL at +50 g and {top2[1][1]:+.2f} mg/dL at -50 g (total range {abs(top2[1][0] - top2[1][1]):.2f} mg/dL).")
            latex_lines.append(f"{top3[0].replace('_', ' ').title()} showed {top3[1][0]:+.2f} mg/dL at +50 g and {top3[1][1]:+.2f} mg/dL at -50 g (total range {abs(top3[1][0] - top3[1][1]):.2f} mg/dL).")
        
        # AZT1D insights
        azt1d_results = all_results[all_results['dataset'] == 'azt1d']
        azt1d_avg = azt1d_results.groupby(['feature', 'increment'])['glucose_change'].mean().reset_index()
        
        azt1d_features = ['carbohydrates', 'insulin', 'correction']
        azt1d_effects = {}
        
        for feature in azt1d_features:
            feature_data = azt1d_avg[azt1d_avg['feature'] == feature]
            if len(feature_data) > 0:
                max_positive = feature_data[feature_data['increment'] == 50]['glucose_change'].iloc[0] if len(feature_data[feature_data['increment'] == 50]) > 0 else 0
                max_negative = feature_data[feature_data['increment'] == -50]['glucose_change'].iloc[0] if len(feature_data[feature_data['increment'] == -50]) > 0 else 0
                azt1d_effects[feature] = (max_positive, max_negative)
        
        azt1d_sorted = sorted(azt1d_effects.items(), key=lambda x: abs(x[1][0] - x[1][1]), reverse=True)
        
        if len(azt1d_sorted) >= 2:
            top1, top2 = azt1d_sorted[:2]
            units = "units" if top1[0] in ["insulin", "correction"] else "g"
            latex_lines.append(f"For AZT1D, {top1[0]} modifications produced {top1[1][0]:+.2f} mg/dL at +50 {units} and {top1[1][1]:+.2f} mg/dL at -50 {units} (total range {abs(top1[1][0] - top1[1][1]):.2f} mg/dL).")
            units2 = "units" if top2[0] in ["insulin", "correction"] else "g"
            latex_lines.append(f"{top2[0].title()} resulted in {top2[1][0]:+.2f} mg/dL at +50 {units2} and {top2[1][1]:+.2f} mg/dL at -50 {units2} (total range {abs(top2[1][0] - top2[1][1]):.2f} mg/dL).")
        
        # Patient variability
        latex_lines.append("Individual patient responses showed considerable variability around the reported average values across both datasets.")
        
        return ' '.join(latex_lines)
    
    # Generate insights
    print("\n" + "="*60)
    print("MACRONUTRIENT MODIFICATION INSIGHTS")
    print("="*60)
    
    insights = analyze_modification_insights(all_results)
    for insight in insights:
        print(f"- {insight}")
    
    # Save insights
    with open('results/macronutrient_modification_insights.txt', 'w') as f:
        f.write('\n'.join(insights))
    
    # Generate LaTeX insights
    latex_insights = generate_latex_insights(all_results)
    with open('results/macronutrient_modification_latex.txt', 'w') as f:
        f.write(latex_insights + '\n')
    
    print(f"\nLaTeX insights: {latex_insights}")
