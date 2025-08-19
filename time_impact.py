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
    
    # Add hour and time features
    df['hour'] = df['datetime'].dt.hour
    df['time'] = df['hour'] + df['datetime'].dt.minute / 60
    
    # Sort by datetime first
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Add prediction horizon features
    for horizon in PREDICTION_HORIZONS:
        df[f'glucose_{horizon}'] = df['glucose'].shift(-horizon) - df['glucose']
    
    # Add glucose change and projected features
    df['glucose_change'] = df['glucose'] - df['glucose'].shift(1)
    df['glucose_change_projected'] = df['glucose_change'].rolling(6, min_periods=6).apply(lambda window: get_projected_value(window, 6))
    df['glucose_projected'] = df['glucose'].rolling(6, min_periods=6).apply(lambda window: get_projected_value(window, 6))
    
    # Drop rows with missing glucose_24 values (similar to azt1d.py)
    df = df.dropna(subset=[f'glucose_24'])
    
    # Keep only needed columns
    glucose_data = df[['patient', 'datetime', 'glucose', 'hour', 'time'] + [f'glucose_{h}' for h in PREDICTION_HORIZONS] + ['glucose_change', 'glucose_change_projected', 'glucose_projected']].copy()
    combined_data = df[['patient', 'datetime', 'carbohydrates', 'insulin', 'correction']].copy()
    
    return glucose_data, combined_data

def modify_time(data, target_hour):
    """Modify the hour component of datetime while keeping other components"""
    modified_data = data.copy()
    modified_data['datetime'] = pd.to_datetime(modified_data['datetime'])
    
    # Set the hour to target_hour while keeping the date and minute/second
    # Replace the hour component with target_hour
    modified_data['datetime'] = modified_data['datetime'].dt.floor('D') + pd.Timedelta(hours=target_hour) + pd.to_timedelta(modified_data['datetime'].dt.minute, unit='m') + pd.to_timedelta(modified_data['datetime'].dt.second, unit='s')
    
    return modified_data

def analyze_time_effect(dataset):
    """Analyze time effect for a given dataset"""
    print(f"Analyzing {dataset.upper()} time effects...")
    
    if dataset == 'd1namo':
        patients = PATIENTS_D1NAMO
        optimization_features = OPTIMIZATION_FEATURES_D1NAMO
        features_to_remove = FEATURES_TO_REMOVE_D1NAMO
        get_data_func = get_d1namo_data
        params_file = 'results/bezier_params/d1namo_all_patient_bezier_params.json'
        training_days = 3
    else:  # azt1d
        patients = PATIENTS_AZT1D
        optimization_features = OPTIMIZATION_FEATURES_AZT1D
        features_to_remove = FEATURES_TO_REMOVE_AZT1D
        get_data_func = get_azt1d_data
        params_file = 'results/bezier_params/azt1d_all_patient_bezier_params.json'
        training_days = 14
    
    # Load all patient Bézier parameters  
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
    
    features_to_remove_full = features_to_remove + [f'glucose_{h}' for h in PREDICTION_HORIZONS] + ['patient_id', 'patient']
    
    # Prepare training data using stepwise retraining approach
    all_train_data = []
    for train_patient in patients:
        train_glucose, train_combined = get_data_func(train_patient)
        train_processed = add_temporal_features(global_params, optimization_features, 
                                            train_glucose, train_combined, DEFAULT_PREDICTION_HORIZON)
        
        if dataset == 'd1namo':
            # Use only training days (first N days)
            train_days = train_processed['datetime'].dt.day.unique()[:training_days]
            train_subset = train_processed[train_processed['datetime'].dt.day.isin(train_days)]
        else:  # azt1d
            # Use first N days normalized
            train_dates = sorted(train_processed['datetime'].dt.normalize().unique())[:training_days]
            train_subset = train_processed[train_processed['datetime'].dt.normalize().isin(train_dates)]
        
        if len(train_subset) > 0:
            all_train_data.append(train_subset)

    # Combine all training data
    X_train_all = pd.concat(all_train_data, ignore_index=True)
    
    # Remove features not used for prediction
    available_features = X_train_all.columns.difference(features_to_remove_full)
    X_train = X_train_all[available_features]
    y_train = X_train_all[f'glucose_{DEFAULT_PREDICTION_HORIZON}']
    
    # Add monotone constraints
    lgb_params_with_constraints = LGB_PARAMS.copy()
    lgb_params_with_constraints['monotone_constraints'] = [MONOTONE_MAP.get(col, 0) for col in available_features]
    lgb_params_with_constraints['callbacks'] = [lgb.early_stopping(10)]
    
    # Train model
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    model = lgb.train(lgb_params_with_constraints, lgb.Dataset(X_tr, label=y_tr), 
                     valid_sets=[lgb.Dataset(X_val, label=y_val)])
    
    time_results = []
    
    for patient in patients:
        print(f"  Processing patient {patient}...")
        
        # Get patient data
        glucose_data, combined_data = get_data_func(patient)
        
        # Get baseline data using original features
        baseline_data = add_temporal_features(global_params, optimization_features, 
                                          glucose_data, combined_data, DEFAULT_PREDICTION_HORIZON)
        
        if dataset == 'd1namo':
            # Use only data from test days (after training days)
            test_days = baseline_data['datetime'].dt.day.unique()[training_days:]
            baseline_test = baseline_data[baseline_data['datetime'].dt.day.isin(test_days)]
        else:  # azt1d
            # Use week after training period
            test_dates = sorted(baseline_data['datetime'].dt.normalize().unique())[training_days:training_days+7]
            baseline_test = baseline_data[baseline_data['datetime'].dt.normalize().isin(test_dates)]
        
        if len(baseline_test) == 0:
            continue
            
        # Get baseline prediction using available data points
        X_baseline = baseline_test[available_features].fillna(0)
        baseline_predictions = model.predict(X_baseline)
        baseline_mean = np.mean(baseline_predictions)
        
        # For each hour of the day, test time modifications
        for hour in range(24):
            # Modify the glucose data time
            modified_glucose = modify_time(glucose_data, hour)
            
            # Re-generate features with modified glucose data
            modified_data = add_temporal_features(global_params, optimization_features, 
                                              modified_glucose, combined_data, DEFAULT_PREDICTION_HORIZON)
            
            if dataset == 'd1namo':
                modified_test = modified_data[modified_data['datetime'].dt.day.isin(test_days)]
            else:  # azt1d
                modified_test = modified_data[modified_data['datetime'].dt.normalize().isin(test_dates)]
            
            if len(modified_test) == 0:
                continue
            
            # Make predictions with modified data
            X_modified = modified_test[available_features].fillna(0)
            modified_predictions = model.predict(X_modified)
            modified_mean = np.mean(modified_predictions)
            
            # Calculate change
            glucose_change = modified_mean - baseline_mean
            
            # Store results
            time_results.append({
                'dataset': dataset,
                'patient': patient,
                'hour': hour,
                'glucose_change': glucose_change,
                'baseline_prediction': baseline_mean,
                'modified_prediction': modified_mean
            })

    return pd.DataFrame(time_results)

def create_combined_visualization(results_df):
    """Create visualization for both dataset time effects"""
    
    # Create figure with subplots for both datasets
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Colors for patients
    d1namo_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    azt1d_colors = plt.cm.tab20(np.linspace(0, 1, len(PATIENTS_AZT1D)))
    
    # D1namo subplot
    d1namo_results = results_df[results_df['dataset'] == 'd1namo']
    if len(d1namo_results) > 0:
        hour_avg_d1namo = d1namo_results.groupby(['patient', 'hour'])['glucose_change'].mean().reset_index()
        
        for i, patient in enumerate(PATIENTS_D1NAMO):
            patient_data = hour_avg_d1namo[hour_avg_d1namo['patient'] == patient].sort_values('hour')
            if len(patient_data) > 0:
                ax1.plot(patient_data['hour'], patient_data['glucose_change'], 
                        marker='o', linewidth=2, color=d1namo_colors[i], label=f'Patient {patient}', alpha=0.8)
        
        # Overall average for D1namo
        overall_avg_d1namo = hour_avg_d1namo.groupby('hour')['glucose_change'].mean().reset_index()
        ax1.plot(overall_avg_d1namo['hour'], overall_avg_d1namo['glucose_change'], 
                'k-', linewidth=3, marker='s', markersize=6, label='Average', alpha=0.9)
        
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax1.set_xticks(range(0, 24, 4))
        ax1.set_xlabel('Hour of Day', fontsize=12)
        ax1.set_ylabel('Change in Predicted Glucose (mg/dL)', fontsize=12)
        ax1.set_title('D1namo Dataset', fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
    
    # AZT1D subplot
    azt1d_results = results_df[results_df['dataset'] == 'azt1d']
    if len(azt1d_results) > 0:
        hour_avg_azt1d = azt1d_results.groupby(['patient', 'hour'])['glucose_change'].mean().reset_index()
        
        for i, patient in enumerate(PATIENTS_AZT1D):
            patient_data = hour_avg_azt1d[hour_avg_azt1d['patient'] == patient].sort_values('hour')
            if len(patient_data) > 0:
                ax2.plot(patient_data['hour'], patient_data['glucose_change'], 
                        marker='o', linewidth=1, color=azt1d_colors[i], alpha=0.7, label=f'Patient {patient}')
        
        # Overall average for AZT1D
        overall_avg_azt1d = hour_avg_azt1d.groupby('hour')['glucose_change'].mean().reset_index()
        ax2.plot(overall_avg_azt1d['hour'], overall_avg_azt1d['glucose_change'], 
                'k-', linewidth=3, marker='s', markersize=6, label='Average', alpha=0.9)
        
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.set_xticks(range(0, 24, 4))
        ax2.set_xlabel('Hour of Day', fontsize=12)
        ax2.set_ylabel('Change in Predicted Glucose (mg/dL)', fontsize=12)
        ax2.set_title('AZT1D Dataset', fontsize=14, fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('manuscript/images/supplementary_data/combined_time_effect_analysis.eps', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate and return statistics for both datasets
    stats = {}
    
    if len(d1namo_results) > 0:
        d1namo_range = overall_avg_d1namo['glucose_change'].max() - overall_avg_d1namo['glucose_change'].min()
        d1namo_peak_hour = overall_avg_d1namo.loc[overall_avg_d1namo['glucose_change'].idxmax(), 'hour']
        d1namo_low_hour = overall_avg_d1namo.loc[overall_avg_d1namo['glucose_change'].idxmin(), 'hour']
        
        d1namo_patient_stats = []
        for patient in PATIENTS_D1NAMO:
            patient_data = hour_avg_d1namo[hour_avg_d1namo['patient'] == patient]
            if len(patient_data) > 0:
                p_range = patient_data['glucose_change'].max() - patient_data['glucose_change'].min()
                p_peak = patient_data.loc[patient_data['glucose_change'].idxmax(), 'hour']
                p_low = patient_data.loc[patient_data['glucose_change'].idxmin(), 'hour']
                d1namo_patient_stats.append({
                    'patient': patient,
                    'range': p_range,
                    'peak_hour': p_peak,
                    'low_hour': p_low
                })
        
        stats['d1namo'] = {
            'overall_avg': overall_avg_d1namo,
            'range': d1namo_range,
            'peak_hour': d1namo_peak_hour,
            'low_hour': d1namo_low_hour,
            'patient_stats': d1namo_patient_stats
        }
    
    if len(azt1d_results) > 0:
        azt1d_range = overall_avg_azt1d['glucose_change'].max() - overall_avg_azt1d['glucose_change'].min()
        azt1d_peak_hour = overall_avg_azt1d.loc[overall_avg_azt1d['glucose_change'].idxmax(), 'hour']
        azt1d_low_hour = overall_avg_azt1d.loc[overall_avg_azt1d['glucose_change'].idxmin(), 'hour']
        
        azt1d_patient_stats = []
        for patient in PATIENTS_AZT1D:
            patient_data = hour_avg_azt1d[hour_avg_azt1d['patient'] == patient]
            if len(patient_data) > 0:
                p_range = patient_data['glucose_change'].max() - patient_data['glucose_change'].min()
                p_peak = patient_data.loc[patient_data['glucose_change'].idxmax(), 'hour']
                p_low = patient_data.loc[patient_data['glucose_change'].idxmin(), 'hour']
                azt1d_patient_stats.append({
                    'patient': patient,
                    'range': p_range,
                    'peak_hour': p_peak,
                    'low_hour': p_low
                })
        
        stats['azt1d'] = {
            'overall_avg': overall_avg_azt1d,
            'range': azt1d_range,
            'peak_hour': azt1d_peak_hour,
            'low_hour': azt1d_low_hour,
            'patient_stats': azt1d_patient_stats
        }
    
    return stats


# Analyze both datasets
print("=== Running Time Impact Analysis for Both Datasets ===\n")

all_results = []

# Analyze D1namo
d1namo_results = analyze_time_effect('d1namo')
all_results.append(d1namo_results)

# Analyze AZT1D
azt1d_results = analyze_time_effect('azt1d')
all_results.append(azt1d_results)

# Combine results
combined_results = pd.concat(all_results, ignore_index=True)

# Create visualization
stats = create_combined_visualization(combined_results)

# Save results
combined_results.to_csv('results/combined_time_effect_results.csv', index=False)
d1namo_results.to_csv('results/d1namo_time_effect_results.csv', index=False)
azt1d_results.to_csv('results/azt1d_time_effect_results.csv', index=False)

# Print summary
print("\n=== Combined Time Effect Analysis Summary ===")
print(f"Total data points analyzed: {len(combined_results)}")

if 'd1namo' in stats:
    d1namo_stats = stats['d1namo']
    print(f"\n=== D1namo Results ===")
    print(f"Patients analyzed: {len(PATIENTS_D1NAMO)}")
    print(f"Data points: {len(d1namo_results)}")
    print(f"Daily glucose variation: {d1namo_stats['range']:.1f} mg/dL")
    print(f"Peak hour: {int(d1namo_stats['peak_hour'])}:00")
    print(f"Lowest hour: {int(d1namo_stats['low_hour'])}:00")
    
    print(f"\nPatient-Specific Results:")
    for patient_stat in d1namo_stats['patient_stats']:
        print(f"  Patient {patient_stat['patient']}: {patient_stat['range']:.1f} mg/dL range, peak at {int(patient_stat['peak_hour'])}:00")

if 'azt1d' in stats:
    azt1d_stats = stats['azt1d']
    print(f"\n=== AZT1D Results ===")
    print(f"Patients analyzed: {len(PATIENTS_AZT1D)}")
    print(f"Data points: {len(azt1d_results)}")
    print(f"Daily glucose variation: {azt1d_stats['range']:.1f} mg/dL")
    print(f"Peak hour: {int(azt1d_stats['peak_hour'])}:00")
    print(f"Lowest hour: {int(azt1d_stats['low_hour'])}:00")
    
    print(f"\nTop 5 Patients with Highest Variation:")
    sorted_patients = sorted(azt1d_stats['patient_stats'], key=lambda x: x['range'], reverse=True)[:5]
    for patient_stat in sorted_patients:
        print(f"  Patient {patient_stat['patient']}: {patient_stat['range']:.1f} mg/dL range, peak at {int(patient_stat['peak_hour'])}:00")
    
    print(f"\nTop 5 Patients with Lowest Variation:")
    sorted_patients_low = sorted(azt1d_stats['patient_stats'], key=lambda x: x['range'])[:5]
    for patient_stat in sorted_patients_low:
        print(f"  Patient {patient_stat['patient']}: {patient_stat['range']:.1f} mg/dL range, peak at {int(patient_stat['peak_hour'])}:00")
