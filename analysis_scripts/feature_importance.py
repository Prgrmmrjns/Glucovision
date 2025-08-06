import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import warnings
import seaborn as sns

warnings.filterwarnings('ignore')

from params import *
from processing_functions import *

# Additional constants specific to this analysis
prediction_horizons_analysis = [6, 12, 24]  # 30min, 60min, 120min
current_patient_weight = 10  # From d1namo.py
validation_size = 0.2

# Load global Bézier parameters from d1namo
with open(f'{RESULTS_PATH}/d1namo_bezier_params.json', 'r') as f:
    global_params = json.load(f)

def analyze_patient_specific_feature_importances():
    """Analyze patient-specific feature importances using d1namo.py weighting approach."""
    
    # Prepare all patient data
    all_data_list = []
    for patient in PATIENTS_D1NAMO:
        glucose_data, combined_data = get_d1namo_data(patient)
        patient_data = add_d1namo_features(global_params, OPTIMIZATION_FEATURES_D1NAMO, (glucose_data, combined_data))
        patient_data['patient_id'] = patient
        all_data_list.append(patient_data)
    
    all_data = pd.concat(all_data_list, ignore_index=True)
    
    # Results storage
    importance_results = []
    
    for prediction_horizon in prediction_horizons_analysis:
        print(f"Analyzing patient-specific feature importances for {prediction_horizon*5}-minute prediction horizon...")
        
        target_feature = f'glucose_{prediction_horizon}'
        
        for target_patient in PATIENTS_D1NAMO:
            # Prepare weighted training data exactly like d1namo.py
            patient_mask = all_data['patient_id'] == target_patient
            
            # Use training days (first 3 days) from all patients
            train_data = []
            for patient in PATIENTS_D1NAMO:
                patient_subset = all_data[all_data['patient_id'] == patient]
                train_days = patient_subset['datetime'].dt.day.unique()[:3]
                train_subset = patient_subset[patient_subset['datetime'].dt.day.isin(train_days)]
                if len(train_subset) > 0:
                    train_data.append(train_subset)
            
            if not train_data:
                continue
                
            # Combine training data
            X_train_all = pd.concat(train_data, ignore_index=True)
            
            # Remove features not used for prediction
            available_features = X_train_all.columns.difference(FEATURES_TO_REMOVE_D1NAMO)
            X_train = X_train_all[available_features]
            y_train = X_train_all[target_feature]
            
            # Create weights: target patient gets weight 10, others get weight 1
            weights = np.where(X_train_all['patient_id'] == target_patient, current_patient_weight, 1)
            
            # Train/validation split with weights
            indices = train_test_split(range(len(X_train)), test_size=validation_size, random_state=42)
            train_idx, val_idx = indices[0], indices[1]
            
            X_tr, y_tr, weights_tr = X_train.iloc[train_idx], y_train.iloc[train_idx], weights[train_idx]
            X_val, y_val, weights_val = X_train.iloc[val_idx], y_train.iloc[val_idx], weights[val_idx]
            
            # Train model with weights
            model = lgb.train(LGB_PARAMS, 
                             lgb.Dataset(X_tr, label=y_tr, weight=weights_tr), 
                             valid_sets=[lgb.Dataset(X_val, label=y_val, weight=weights_val)])
            
            # Get feature importances
            feature_names = model.feature_name()
            importances = model.feature_importance(importance_type='gain')
            
            # Normalize to percentages
            total_importance = sum(importances)
            importance_percentages = [(imp / total_importance) * 100 for imp in importances]
            
            # Store results
            for feature_name, importance_pct in zip(feature_names, importance_percentages):
                importance_results.append({
                    'patient': target_patient,
                    'prediction_horizon_minutes': prediction_horizon * 5,
                    'prediction_horizon': prediction_horizon,
                    'feature': feature_name,
                    'importance_percentage': importance_pct
                })
    
    return pd.DataFrame(importance_results)

def create_patient_specific_visualization(importance_df):
    """Create visualization of patient-specific feature importances."""
    
    # Focus on 60-minute horizon for main visualization
    df_60 = importance_df[importance_df['prediction_horizon_minutes'] == 60]
    
    # Create heatmap of patient-specific importances
    pivot_data = df_60.pivot(index='feature', columns='patient', values='importance_percentage')
    
    # Sort rows by average feature importance (highest to lowest)
    pivot_data = pivot_data.loc[pivot_data.mean(axis=1).sort_values(ascending=False).index]
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlBu_r', center=pivot_data.mean().mean(),
                cbar_kws={'label': 'Feature Importance (%)'})
    plt.xlabel('Patient', fontsize=12)
    plt.ylabel('')
    plt.yticks(rotation=30, fontsize=8)
    plt.tight_layout()
    plt.savefig('../manuscript/images/results/patient_specific_feature_importances.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_patient_differences(importance_df):
    """Analyze differences in patient-specific feature importances."""
    
    # Focus on 60-minute horizon
    df_60 = importance_df[importance_df['prediction_horizon_minutes'] == 60]
    
    # Calculate patient variations for each feature
    feature_variations = {}
    
    for feature in df_60['feature'].unique():
        feature_data = df_60[df_60['feature'] == feature]
        if len(feature_data) >= len(PATIENTS_D1NAMO):  # Ensure we have data for all patients
            max_patient = feature_data.loc[feature_data['importance_percentage'].idxmax()]
            min_patient = feature_data.loc[feature_data['importance_percentage'].idxmin()]
            
            feature_variations[feature] = {
                'max_patient': max_patient['patient'],
                'max_importance': max_patient['importance_percentage'],
                'min_patient': min_patient['patient'],
                'min_importance': min_patient['importance_percentage'],
                'range': max_patient['importance_percentage'] - min_patient['importance_percentage'],
                'std': feature_data['importance_percentage'].std()
            }
    
    # Find features with highest patient variation
    high_variation_features = sorted(feature_variations.items(), 
                                   key=lambda x: x[1]['std'], reverse=True)[:5]
    
    return feature_variations, high_variation_features

# Run analysis
importance_df = analyze_patient_specific_feature_importances()

# Create visualizations
create_patient_specific_visualization(importance_df)

# Analyze patterns
feature_variations, high_variation_features = analyze_patient_differences(importance_df)

# Print results
print("\nPatient-Specific Feature Importance Analysis Results:")
print("=" * 60)

# Calculate average importances by horizon
for horizon in [30, 60, 120]:
    horizon_data = importance_df[importance_df['prediction_horizon_minutes'] == horizon]
    avg_by_feature = horizon_data.groupby('feature')['importance_percentage'].mean().sort_values(ascending=False)
    
    print(f"\n{horizon}-minute Prediction Horizon (Patient-Weighted):")
    print(f"  Top 5 features:")
    for i, (feature, importance) in enumerate(avg_by_feature.head(5).items()):
        print(f"    {i+1}. {feature}: {importance:.1f}%")

print(f"\nFeatures with highest patient variation (60-minute horizon):")
for feature, data in high_variation_features:
    print(f"  {feature}: Patient {data['max_patient']} ({data['max_importance']:.1f}%) vs Patient {data['min_patient']} ({data['min_importance']:.1f}%) [std: {data['std']:.1f}%]")

# Print specific interesting patient differences
df_60 = importance_df[importance_df['prediction_horizon_minutes'] == 60]

print(f"\nNotable patient-specific patterns:")
for feature in ['complex_sugars', 'simple_sugars', 'fats', 'time']:
    if feature in df_60['feature'].values:
        feature_data = df_60[df_60['feature'] == feature]
        max_row = feature_data.loc[feature_data['importance_percentage'].idxmax()]
        min_row = feature_data.loc[feature_data['importance_percentage'].idxmin()]
        
        print(f"  {feature.replace('_', ' ').title()}: Patient {max_row['patient']} ({max_row['importance_percentage']:.1f}%) vs Patient {min_row['patient']} ({min_row['importance_percentage']:.1f}%)")

# Save results
importance_df.to_csv(f'{RESULTS_PATH}/patient_specific_feature_importance_analysis.csv', index=False)