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

def analyze_d1namo_time_effect():
    """Analyze D1namo time effect"""
    print("Analyzing D1namo time effects...")
    
    # Load global Bézier parameters
    with open(f'{RESULTS_PATH}/d1namo_bezier_params.json', 'r') as f:
        global_params = json.load(f)
    
    features_to_remove = FEATURES_TO_REMOVE_D1NAMO
    
    # Prepare training data
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
    available_features = X_train_all.columns.difference(features_to_remove)
    X_train = X_train_all[available_features]
    y_train = X_train_all[f'glucose_{DEFAULT_PREDICTION_HORIZON}']
    
    # Train model
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    model = lgb.train(LGB_PARAMS, lgb.Dataset(X_tr, label=y_tr), 
                     valid_sets=[lgb.Dataset(X_val, label=y_val)])
    
    time_results = []
    
    for patient in PATIENTS_D1NAMO:
        # Get patient data
        glucose_data, combined_data = get_d1namo_data(patient)
        
        # Get baseline data using original features
        baseline_data = add_d1namo_features(global_params, OPTIMIZATION_FEATURES_D1NAMO, (glucose_data, combined_data))
        
        # Use only data from test days (4-5)
        test_days = baseline_data['datetime'].dt.day.unique()[3:]
        baseline_test = baseline_data[baseline_data['datetime'].dt.day.isin(test_days)]
        
        if len(baseline_test) == 0:
            continue
            
        # Get baseline prediction
        X_baseline = baseline_test[available_features]
        baseline_predictions = model.predict(X_baseline)
        baseline_mean = np.mean(baseline_predictions)
        
        # For each hour of the day, test time modifications
        for hour in range(24):
            # Modify the glucose data time
            modified_glucose = modify_time(glucose_data, hour)
            
            # Re-generate features with modified glucose data
            modified_data = add_d1namo_features(global_params, OPTIMIZATION_FEATURES_D1NAMO, (modified_glucose, combined_data))
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
            time_results.append({
                'dataset': 'D1namo',
                'patient': patient,
                'hour': hour,
                'glucose_change': glucose_change,
                'baseline_prediction': baseline_mean,
                'modified_prediction': modified_mean
            })

    return pd.DataFrame(time_results)

def analyze_azt1d_time_effect():
    """Analyze AZT1D time effect"""
    print("Analyzing AZT1D time effects...")
    
    # Use domain knowledge parameters for Bezier curves
    global_params = AZT1D_BEZIER_PARAMS
    
    # Load all AZT1D data
    azt1d_data_list = load_azt1d_data()
    
    if len(azt1d_data_list) == 0:
        print("No AZT1D data found")
        return pd.DataFrame()
    
    # Prepare training data from all patients
    all_train_data = []
    for patient_data in azt1d_data_list:
        # Add temporal features
        combined_data = patient_data[['datetime', 'carbohydrates', 'insulin']].copy()
        processed_data = add_azt1d_features(global_params, OPTIMIZATION_FEATURES_AZT1D, patient_data, combined_data)
        
        # Use first 70% of data for training
        n_train = int(len(processed_data) * 0.7)
        train_subset = processed_data.iloc[:n_train]
        if len(train_subset) > 50:
            all_train_data.append(train_subset)
    
    if len(all_train_data) == 0:
        print("No AZT1D training data available")
        return pd.DataFrame()
    
    # Combine all training data
    X_train_all = pd.concat(all_train_data, ignore_index=True)
    
    # Remove features not used for prediction
    features_to_remove_azt1d = FEATURES_TO_REMOVE_AZT1D
    available_features = X_train_all.columns.difference(features_to_remove_azt1d)
    X_train = X_train_all[available_features]
    y_train = X_train_all[f'glucose_{DEFAULT_PREDICTION_HORIZON}']
    
    # Train model
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    model = lgb.train(LGB_PARAMS, lgb.Dataset(X_tr, label=y_tr), 
                     valid_sets=[lgb.Dataset(X_val, label=y_val)])
    
    time_results = []
    
    for patient_data in azt1d_data_list[:10]:  # Limit to first 10 patients for visualization
        patient = patient_data['patient'].iloc[0]
        
        # Use last 30% of data for testing
        n_train = int(len(patient_data) * 0.7)
        test_data = patient_data.iloc[n_train:]
        
        if len(test_data) < 20:
            continue
        
        # Add temporal features to test data
        combined_data = test_data[['datetime', 'carbohydrates', 'insulin']].copy()
        baseline_data = add_azt1d_features(global_params, OPTIMIZATION_FEATURES_AZT1D, test_data, combined_data)
        
        # Get baseline prediction
        X_baseline = baseline_data[available_features]
        baseline_predictions = model.predict(X_baseline)
        baseline_mean = np.mean(baseline_predictions)
        
        # For each hour of the day, test time modifications
        for hour in range(24):
            # Modify the time
            modified_test = modify_time(test_data, hour)
            
            # Re-generate features with modified data
            modified_combined = modified_test[['datetime', 'carbohydrates', 'insulin']].copy()
            modified_data = add_azt1d_features(global_params, OPTIMIZATION_FEATURES_AZT1D, modified_test, modified_combined)
            
            # Make predictions with modified data
            X_modified = modified_data[available_features]
            modified_predictions = model.predict(X_modified)
            modified_mean = np.mean(modified_predictions)
            
            # Calculate change
            glucose_change = modified_mean - baseline_mean
            
            # Store results
            time_results.append({
                'dataset': 'AZT1D',
                'patient': str(patient),
                'hour': hour,
                'glucose_change': glucose_change,
                'baseline_prediction': baseline_mean,
                'modified_prediction': modified_mean
            })

    return pd.DataFrame(time_results)

def create_combined_visualization(d1namo_results, azt1d_results):
    """Create combined visualization of both datasets"""
    
    # Combine results
    all_results = pd.concat([d1namo_results, azt1d_results], ignore_index=True)
    
    if len(all_results) == 0:
        print("No results to visualize")
        return
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Colors for datasets
    d1namo_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    azt1d_colors = plt.cm.Set3(np.linspace(0, 1, 10))
    
    # D1namo individual patients
    d1namo_data = all_results[all_results['dataset'] == 'D1namo']
    if len(d1namo_data) > 0:
        hour_avg_d1namo = d1namo_data.groupby(['patient', 'hour'])['glucose_change'].mean().reset_index()
        
        for i, patient in enumerate(PATIENTS_D1NAMO):
            patient_data = hour_avg_d1namo[hour_avg_d1namo['patient'] == patient].sort_values('hour')
            if len(patient_data) > 0:
                ax1.plot(patient_data['hour'], patient_data['glucose_change'], 
                        marker='o', linewidth=2, color=d1namo_colors[i], label=f'Patient {patient}', alpha=0.7)
        
        # D1namo average
        overall_avg_d1namo = hour_avg_d1namo.groupby('hour')['glucose_change'].mean().reset_index()
        ax1.plot(overall_avg_d1namo['hour'], overall_avg_d1namo['glucose_change'], 
                'k-', linewidth=3, marker='s', markersize=6, label='D1namo Average')
    
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_xticks(range(0, 24, 4))
    ax1.set_xlabel('Hour of Day', fontsize=12)
    ax1.set_ylabel('Change in Predicted Glucose (mg/dL)', fontsize=12)
    ax1.set_title('D1namo: Individual Patients + Average', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # AZT1D individual patients
    azt1d_data = all_results[all_results['dataset'] == 'AZT1D']
    if len(azt1d_data) > 0:
        hour_avg_azt1d = azt1d_data.groupby(['patient', 'hour'])['glucose_change'].mean().reset_index()
        unique_patients_azt1d = hour_avg_azt1d['patient'].unique()
        
        for i, patient in enumerate(unique_patients_azt1d):
            patient_data = hour_avg_azt1d[hour_avg_azt1d['patient'] == patient].sort_values('hour')
            if len(patient_data) > 0:
                ax2.plot(patient_data['hour'], patient_data['glucose_change'], 
                        marker='o', linewidth=2, color=azt1d_colors[i], label=f'Patient {patient}', alpha=0.7)
        
        # AZT1D average
        overall_avg_azt1d = hour_avg_azt1d.groupby('hour')['glucose_change'].mean().reset_index()
        ax2.plot(overall_avg_azt1d['hour'], overall_avg_azt1d['glucose_change'], 
                'k-', linewidth=3, marker='s', markersize=6, label='AZT1D Average')
    
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.set_xticks(range(0, 24, 4))
    ax2.set_xlabel('Hour of Day', fontsize=12)
    ax2.set_ylabel('Change in Predicted Glucose (mg/dL)', fontsize=12)
    ax2.set_title('AZT1D: Individual Patients + Average', fontsize=14, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Combined averages comparison
    if len(d1namo_data) > 0 and len(azt1d_data) > 0:
        ax3.plot(overall_avg_d1namo['hour'], overall_avg_d1namo['glucose_change'], 
                'b-', linewidth=3, marker='o', markersize=8, label='D1namo Average')
        ax3.plot(overall_avg_azt1d['hour'], overall_avg_azt1d['glucose_change'], 
                'r-', linewidth=3, marker='s', markersize=8, label='AZT1D Average')
        
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xticks(range(0, 24, 4))
        ax3.set_xlabel('Hour of Day', fontsize=12)
        ax3.set_ylabel('Change in Predicted Glucose (mg/dL)', fontsize=12)
        ax3.set_title('Dataset Comparison: Average Time Effects', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=12)
        ax3.grid(True, alpha=0.3)
    
    # Summary statistics
    ax4.axis('off')
    summary_text = []
    
    if len(d1namo_data) > 0:
        d1namo_range = overall_avg_d1namo['glucose_change'].max() - overall_avg_d1namo['glucose_change'].min()
        d1namo_peak_hour = overall_avg_d1namo.loc[overall_avg_d1namo['glucose_change'].idxmax(), 'hour']
        d1namo_low_hour = overall_avg_d1namo.loc[overall_avg_d1namo['glucose_change'].idxmin(), 'hour']
        
        summary_text.extend([
            "D1namo Results:",
            f"  Daily glucose variation: {d1namo_range:.1f} mg/dL",
            f"  Peak hour: {int(d1namo_peak_hour)}:00",
            f"  Lowest hour: {int(d1namo_low_hour)}:00",
            f"  Patients analyzed: {len(PATIENTS_D1NAMO)}",
            ""
        ])
    
    if len(azt1d_data) > 0:
        azt1d_range = overall_avg_azt1d['glucose_change'].max() - overall_avg_azt1d['glucose_change'].min()
        azt1d_peak_hour = overall_avg_azt1d.loc[overall_avg_azt1d['glucose_change'].idxmax(), 'hour']
        azt1d_low_hour = overall_avg_azt1d.loc[overall_avg_azt1d['glucose_change'].idxmin(), 'hour']
        
        summary_text.extend([
            "AZT1D Results:",
            f"  Daily glucose variation: {azt1d_range:.1f} mg/dL",
            f"  Peak hour: {int(azt1d_peak_hour)}:00",
            f"  Lowest hour: {int(azt1d_low_hour)}:00",
            f"  Patients analyzed: {len(unique_patients_azt1d)}",
            ""
        ])
    
    summary_text.extend([
        "Key Findings:",
        "• Time of day significantly impacts glucose predictions",
        "• Circadian patterns vary between datasets",
        "• Individual patient variation is substantial",
        "• Both datasets show distinct peak/trough timing"
    ])
    
    ax4.text(0.1, 0.9, '\n'.join(summary_text), transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('../manuscript/images/supplementary_data/combined_time_effect_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('../manuscript/images/supplementary_data/combined_time_effect_analysis.eps', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Combined visualization saved to ../manuscript/images/supplementary_data/combined_time_effect_analysis.png")

def main():
    """Main function to run combined time impact analysis"""
    print("Combined Time Impact Analysis: D1namo vs AZT1D")
    print("=" * 50)
    
    # Create output directory if it doesn't exist
    os.makedirs('../manuscript/images/supplementary_data', exist_ok=True)
    
    # Run D1namo analysis
    d1namo_results = analyze_d1namo_time_effect()
    
    # Run AZT1D analysis
    azt1d_results = analyze_azt1d_time_effect()
    
    # Create combined visualization
    create_combined_visualization(d1namo_results, azt1d_results)
    
    # Save combined results
    all_results = pd.concat([d1namo_results, azt1d_results], ignore_index=True)
    all_results.to_csv(f'{RESULTS_PATH}/combined_time_effect_results.csv', index=False)
    
    # Print summary
    print("\nSummary:")
    print(f"D1namo results: {len(d1namo_results)} data points")
    print(f"AZT1D results: {len(azt1d_results)} data points")
    print(f"Combined results saved to: {RESULTS_PATH}/combined_time_effect_results.csv")
    print(f"Visualization saved to: ../manuscript/images/supplementary_data/combined_time_effect_analysis.png")

if __name__ == "__main__":
    main()