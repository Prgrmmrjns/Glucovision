import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

from params import *
from processing_functions import *

# Load global Bézier parameters for both datasets
with open('results/bezier_params/d1namo_all_patient_bezier_params.json', 'r') as f:
    d1namo_all_patient_params = json.load(f)

with open('results/bezier_params/azt1d_all_patient_bezier_params.json', 'r') as f:
    azt1d_all_patient_params = json.load(f)

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

d1namo_params = calculate_global_params(d1namo_all_patient_params, OPTIMIZATION_FEATURES_D1NAMO)
azt1d_params = calculate_global_params(azt1d_all_patient_params, OPTIMIZATION_FEATURES_AZT1D)

def analyze_overall_feature_importances(dataset_name, params, optimization_features, patients, load_data_func):
    """Analyze overall feature importances across all patients for a dataset."""
    
    # Prepare all patient data
    all_data_list = []
    successful_patients = []
    
    for patient in patients:
        glucose_data, combined_data = load_data_func(patient)
        patient_data = add_temporal_features(params, optimization_features, glucose_data, combined_data, prediction_horizon=0)
        patient_data['patient_id'] = patient
        all_data_list.append(patient_data)
        successful_patients.append(patient)
    
    all_data = pd.concat(all_data_list, ignore_index=True)
    
    # Results storage
    importance_results = []
    
    for prediction_horizon in PREDICTION_HORIZONS:
        
        target_feature = f'glucose_{prediction_horizon}'
        
        # Use training days (first 3 or 14 days depending on dataset) from all patients
        train_data = []
        training_days = 14 if dataset_name == 'AZT1D' else 3
        
        for patient in patients:
            patient_subset = all_data[all_data['patient_id'] == patient]
            if dataset_name == 'AZT1D':
                train_dates = sorted(patient_subset['datetime'].dt.normalize().unique())[:training_days]
                train_subset = patient_subset[patient_subset['datetime'].dt.normalize().isin(train_dates)]
            else:
                train_days = patient_subset['datetime'].dt.day.unique()[:training_days]
                train_subset = patient_subset[patient_subset['datetime'].dt.day.isin(train_days)]
            
            train_data.append(train_subset)
            
        # Combine training data
        X_train_all = pd.concat(train_data, ignore_index=True)
        
        # Remove features not used for prediction
        if dataset_name == 'AZT1D':
            features_to_remove = FEATURES_TO_REMOVE_AZT1D + ['patient_id']
        else:
            features_to_remove = FEATURES_TO_REMOVE_D1NAMO + ['patient_id']
        
        available_features = X_train_all.columns.difference(features_to_remove)
        feature_cols = [c for c in available_features if X_train_all[c].dtype != 'O']
        
        X_train = X_train_all[feature_cols]
        y_train = X_train_all[target_feature]
        
        # Train/validation split
        train_idx, val_idx = train_test_split(range(len(X_train)), test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)
        
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        # Train model with monotone constraints
        lgb_params = LGB_PARAMS.copy()
        lgb_params['monotone_constraints'] = [MONOTONE_MAP.get(col, 0) for col in feature_cols]
        
        model = lgb.train(lgb_params, 
                         lgb.Dataset(X_tr, label=y_tr), 
                         valid_sets=[lgb.Dataset(X_val, label=y_val)])
        
        # Get feature importances
        feature_names = model.feature_name()
        importances = model.feature_importance(importance_type='gain')
        
        # Normalize to percentages
        total_importance = sum(importances)
        importance_percentages = [(imp / total_importance) * 100 for imp in importances]
        
        # Store results
        for feature_name, importance_pct in zip(feature_names, importance_percentages):
            importance_results.append({
                'dataset': dataset_name,
                'prediction_horizon_minutes': prediction_horizon * 5,
                'prediction_horizon': prediction_horizon,
                'feature': feature_name,
                'importance_percentage': importance_pct
            })
    
    return pd.DataFrame(importance_results)

def create_cumulative_bar_chart(importance_df):
    """Create cumulative bar chart visualization for feature importances across datasets and horizons."""
    
    # Feature mapping for better display names
    feature_names = {
        'glucose': 'Glucose',
        'glucose_change': 'Glucose Change', 
        'glucose_projected': 'Glucose Projected',
        'glucose_change_projected': 'Glucose Change Projected',
        'time': 'Time of Day',
        'simple_sugars': 'Simple Sugars',
        'complex_sugars': 'Complex Sugars',
        'proteins': 'Proteins',
        'fats': 'Fats',
        'dietary_fibers': 'Dietary Fibers',
        'insulin': 'Insulin',
        'carbohydrates': 'Carbohydrates',
        'correction': 'Correction Insulin'
    }
    
    # Create color mapping for consistent feature colors across datasets
    all_features = sorted(importance_df['feature'].unique())
    feature_colors = {}
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#DDA0DD', '#87CEEB', '#F0E68C', '#D3D3D3', '#FFA07A', '#98D8C8']
    for i, feature in enumerate(all_features):
        feature_colors[feature] = colors[i % len(colors)]
    
    # Define horizon names
    horizon_names = {6: "30 min", 9: "45 min", 12: "60 min", 18: "90 min", 24: "120 min"}
    
    # Create figure with subplots for each dataset
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Collect all features that will be shown for shared legend
    legend_features = set()
    
    for i, dataset in enumerate(['D1namo', 'AZT1D']):
        ax = axes[i]
        dataset_data = importance_df[importance_df['dataset'] == dataset]
        
        # Get top features across all horizons for this dataset
        top_features = dataset_data.groupby('feature')['importance_percentage'].mean().sort_values(ascending=False).head(8).index
        legend_features.update(top_features)
        
        # Prepare data for stacked bar chart
        horizons = sorted(dataset_data['prediction_horizon'].unique())
        bottom_values = np.zeros(len(horizons))
        
        for feature in top_features:
            feature_data = dataset_data[dataset_data['feature'] == feature]
            values = []
            for horizon in horizons:
                horizon_data = feature_data[feature_data['prediction_horizon'] == horizon]
                values.append(horizon_data['importance_percentage'].iloc[0] if len(horizon_data) > 0 else 0)
            
            display_name = feature_names.get(feature, feature.replace('_', ' ').title())
            ax.bar([horizon_names[h] for h in horizons], values, bottom=bottom_values, 
                   label=display_name, color=feature_colors[feature], alpha=0.8)
            bottom_values += values
        
        ax.set_title(f'{dataset} Dataset', fontsize=14, fontweight='bold')
        ax.set_ylabel('Feature Importance (%)', fontsize=12)
        ax.set_xlabel('Prediction Horizon', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 100)
    
    # Order legend features by overall D1namo importance
    d1namo_data = importance_df[importance_df['dataset'] == 'D1namo']
    d1namo_importance = d1namo_data.groupby('feature')['importance_percentage'].mean().sort_values(ascending=False)
    
    # Separate D1namo features (ordered by importance) and AZT1D-only features
    d1namo_features = [f for f in d1namo_importance.index if f in legend_features]
    azt1d_only_features = sorted([f for f in legend_features if f not in d1namo_importance.index])
    ordered_features = d1namo_features + azt1d_only_features
    
    # Create shared legend with features ordered by D1namo importance
    handles, labels = [], []
    for feature in ordered_features:
        display_name = feature_names.get(feature, feature.replace('_', ' ').title())
        handles.append(plt.Rectangle((0,0),1,1, color=feature_colors[feature], alpha=0.8))
        labels.append(display_name)
    
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.5), fontsize=10)
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)  # Make room for shared legend
    plt.savefig('manuscript/images/results/feature_importances.eps', dpi=300, bbox_inches='tight')
    plt.close()

def load_azt1d_patient_data(patient):
    """Load AZT1D data for a specific patient number using exact same processing as azt1d.py"""
    # Convert patient number to match AZT1D file format
    patient_num = int(patient) if isinstance(patient, str) else patient
    
    # Load AZT1D data
    file_path = f"{AZT1D_DATA_PATH}/Subject {patient_num}/Subject {patient_num}.csv"
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None, None
    
    # Process exactly like azt1d.py
    df['patient'] = patient_num
    df['datetime'] = pd.to_datetime(df[AZT1D_COLUMNS['datetime']])
    df['glucose'] = df[AZT1D_COLUMNS['glucose']].fillna(0)
    df['carbohydrates'] = df[AZT1D_COLUMNS['carbohydrates']].fillna(0)
    df['insulin'] = df[AZT1D_COLUMNS['insulin']].fillna(0)
    df['correction'] = df[AZT1D_COLUMNS['correction']].fillna(0)
    
    # Add hour and time features
    df['hour'] = df['datetime'].dt.hour
    df['time'] = df['hour'] + df['datetime'].dt.minute / 60
    
    # Keep only needed columns
    df = df[['patient', 'datetime', 'glucose', 'carbohydrates', 'insulin', 'correction', 'hour', 'time']].copy()
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Add prediction horizon features
    for horizon in PREDICTION_HORIZONS:
        df[f'glucose_{horizon}'] = df['glucose'].shift(-horizon) - df['glucose']
    
    # Add glucose change and projected features (exactly like azt1d.py)
    df['glucose_change'] = df['glucose'] - df['glucose'].shift(1)
    df['glucose_change_projected'] = df['glucose_change'].rolling(6, min_periods=6).apply(lambda window: get_projected_value(window, 6))
    df['glucose_projected'] = df['glucose'].rolling(6, min_periods=6).apply(lambda window: get_projected_value(window, 6))
    df = df.dropna(subset=[f'glucose_24'])
    
    # Create combined data for temporal features
    combined_data = df[['datetime', 'carbohydrates', 'insulin', 'correction']].copy()
    
    return df, combined_data

# Get AZT1D patient list
azt1d_patients = PATIENTS_AZT1D

# Run analysis for both datasets
d1namo_results = analyze_overall_feature_importances('D1namo', d1namo_params, OPTIMIZATION_FEATURES_D1NAMO, 
                                                     PATIENTS_D1NAMO, get_d1namo_data)

azt1d_results = analyze_overall_feature_importances('AZT1D', azt1d_params, OPTIMIZATION_FEATURES_AZT1D, 
                                                    azt1d_patients, load_azt1d_patient_data)

# Combine results
all_results = pd.concat([d1namo_results, azt1d_results], ignore_index=True)

# Create visualizations
create_cumulative_bar_chart(all_results)

# Print results
print("\nOverall Feature Importance Analysis Results:")
print("=" * 60)

# Calculate average importances by dataset and horizon
for dataset in ['D1namo', 'AZT1D']:
    print(f"\n{dataset} Dataset:")
    dataset_data = all_results[all_results['dataset'] == dataset]
    
    for horizon in PREDICTION_HORIZONS:
        horizon_data = dataset_data[dataset_data['prediction_horizon'] == horizon]
        if len(horizon_data) > 0:
            avg_by_feature = horizon_data.groupby('feature')['importance_percentage'].mean().sort_values(ascending=False)
            
            print(f"\n  {horizon*5}-minute Prediction Horizon:")
            print(f"    Top 5 features:")
            for i, (feature, importance) in enumerate(avg_by_feature.head(5).items()):
                print(f"      {i+1}. {feature.replace('_', ' ').title()}: {importance:.1f}%")

# Print cross-dataset comparison for common features
print("\nCross-dataset feature comparison (60-minute horizon):")
print("-" * 50)
d1namo_60 = all_results[(all_results['dataset'] == 'D1namo') & (all_results['prediction_horizon'] == 12)]
azt1d_60 = all_results[(all_results['dataset'] == 'AZT1D') & (all_results['prediction_horizon'] == 12)]

common_features = ['glucose', 'glucose_change', 'time', 'insulin']
for feature in common_features:
    d1n_imp = d1namo_60[d1namo_60['feature'] == feature]['importance_percentage']
    azt_imp = azt1d_60[azt1d_60['feature'] == feature]['importance_percentage']
    
    d1n_val = d1n_imp.iloc[0] if len(d1n_imp) > 0 else 0
    azt_val = azt_imp.iloc[0] if len(azt_imp) > 0 else 0
    
    print(f"  {feature.replace('_', ' ').title()}: D1namo {d1n_val:.1f}% vs AZT1D {azt_val:.1f}%")

# Save results
all_results.to_csv('results/feature_importance_analysis.csv', index=False)

# ==========================
# Feature Importance Insights
# ==========================

def analyze_feature_importance_insights(all_results):
    """Analyze feature importance patterns and generate insights"""
    
    insights = []
    
    # 1. Temporal evolution analysis
    for dataset in ['D1namo', 'AZT1D']:
        dataset_data = all_results[all_results['dataset'] == dataset]
        
        # Find features that increase/decrease in importance over time
        for feature in dataset_data['feature'].unique():
            feature_data = dataset_data[dataset_data['feature'] == feature]
            if len(feature_data) >= 2:
                horizons = sorted(feature_data['prediction_horizon'].unique())
                if len(horizons) >= 2:
                    first_imp = feature_data[feature_data['prediction_horizon'] == horizons[0]]['importance_percentage'].iloc[0]
                    last_imp = feature_data[feature_data['prediction_horizon'] == horizons[-1]]['importance_percentage'].iloc[0]
                    change = last_imp - first_imp
                    
                    if abs(change) > 5:  # Significant change threshold
                        direction = "increases" if change > 0 else "decreases"
                        insights.append(f"{dataset} {feature}: {direction} from {first_imp:.1f}% to {last_imp:.1f}% ({change:+.1f}%)")
    
    # 2. Cross-dataset comparison at 60 minutes
    d1namo_60 = all_results[(all_results['dataset'] == 'D1namo') & (all_results['prediction_horizon'] == 12)]
    azt1d_60 = all_results[(all_results['dataset'] == 'AZT1D') & (all_results['prediction_horizon'] == 12)]
    
    common_features = ['glucose', 'glucose_change', 'time', 'insulin']
    for feature in common_features:
        d1n_imp = d1namo_60[d1namo_60['feature'] == feature]['importance_percentage']
        azt_imp = azt1d_60[azt1d_60['feature'] == feature]['importance_percentage']
        
        if len(d1n_imp) > 0 and len(azt_imp) > 0:
            d1n_val = d1n_imp.iloc[0]
            azt_val = azt_imp.iloc[0]
            diff = d1n_val - azt_val
            
            if abs(diff) > 5:  # Significant difference threshold
                insights.append(f"60min {feature}: D1namo {d1n_val:.1f}% vs AZT1D {azt_val:.1f}% (diff: {diff:+.1f}%)")
    
    # 3. Macronutrient feature analysis for D1namo
    d1namo_macro_features = ['simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers']
    d1namo_60_macro = d1namo_60[d1namo_60['feature'].isin(d1namo_macro_features)]
    
    if len(d1namo_60_macro) > 0:
        top_macro = d1namo_60_macro.loc[d1namo_60_macro['importance_percentage'].idxmax()]
        insights.append(f"D1namo 60min top macronutrient: {top_macro['feature']} ({top_macro['importance_percentage']:.1f}%)")
    
    # 4. Glucose dynamics dominance analysis
    glucose_features = ['glucose', 'glucose_change', 'glucose_projected', 'glucose_change_projected']
    
    for dataset in ['D1namo', 'AZT1D']:
        dataset_data = all_results[all_results['dataset'] == dataset]
        for horizon in [6, 12, 24]:  # 30, 60, 120 minutes
            horizon_data = dataset_data[dataset_data['prediction_horizon'] == horizon]
            glucose_imp = horizon_data[horizon_data['feature'].isin(glucose_features)]['importance_percentage'].sum()
            total_imp = horizon_data['importance_percentage'].sum()
            glucose_pct = (glucose_imp / total_imp) * 100 if total_imp > 0 else 0
            
            insights.append(f"{dataset} {horizon*5}min: Glucose features {glucose_pct:.1f}% of total importance")
    
    return insights

def generate_latex_insights(all_results):
    """Generate LaTeX-ready insights paragraph"""
    
    # Key findings for LaTeX
    latex_lines = []
    
    # 1. Overall patterns
    d1namo_30 = all_results[(all_results['dataset'] == 'D1namo') & (all_results['prediction_horizon'] == 6)]
    d1namo_120 = all_results[(all_results['dataset'] == 'D1namo') & (all_results['prediction_horizon'] == 24)]
    azt1d_30 = all_results[(all_results['dataset'] == 'AZT1D') & (all_results['prediction_horizon'] == 6)]
    azt1d_120 = all_results[(all_results['dataset'] == 'AZT1D') & (all_results['prediction_horizon'] == 24)]
    
    # Glucose change dominance at 30 minutes
    if len(d1namo_30) > 0:
        glucose_change_30 = d1namo_30[d1namo_30['feature'] == 'glucose_change']['importance_percentage']
        if len(glucose_change_30) > 0:
            latex_lines.append(f"At 30 minutes, glucose change dominates predictions in D1namo ({glucose_change_30.iloc[0]:.1f}%)")
    
    # Time feature evolution
    if len(d1namo_30) > 0 and len(d1namo_120) > 0:
        time_30 = d1namo_30[d1namo_30['feature'] == 'time']['importance_percentage']
        time_120 = d1namo_120[d1namo_120['feature'] == 'time']['importance_percentage']
        if len(time_30) > 0 and len(time_120) > 0:
            latex_lines.append(f"Time of day importance increases from {time_30.iloc[0]:.1f}% at 30 minutes to {time_120.iloc[0]:.1f}% at 120 minutes in D1namo")
    
    # Macronutrient features at 120 minutes
    macro_features = ['simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers']
    if len(d1namo_120) > 0:
        macro_imp = d1namo_120[d1namo_120['feature'].isin(macro_features)]['importance_percentage'].sum()
        latex_lines.append(f"Macronutrient features collectively contribute {macro_imp:.1f}% at 120 minutes in D1namo")
    
    # AZT1D patterns
    if len(azt1d_30) > 0 and len(azt1d_120) > 0:
        glucose_30 = azt1d_30[azt1d_30['feature'] == 'glucose']['importance_percentage']
        glucose_120 = azt1d_120[azt1d_120['feature'] == 'glucose']['importance_percentage']
        if len(glucose_30) > 0 and len(glucose_120) > 0:
            latex_lines.append(f"In AZT1D, current glucose importance increases from {glucose_30.iloc[0]:.1f}% at 30 minutes to {glucose_120.iloc[0]:.1f}% at 120 minutes")
    
    # Cross-dataset comparison
    d1namo_60 = all_results[(all_results['dataset'] == 'D1namo') & (all_results['prediction_horizon'] == 12)]
    azt1d_60 = all_results[(all_results['dataset'] == 'AZT1D') & (all_results['prediction_horizon'] == 12)]
    
    if len(d1namo_60) > 0 and len(azt1d_60) > 0:
        time_d1 = d1namo_60[d1namo_60['feature'] == 'time']['importance_percentage']
        time_azt = azt1d_60[azt1d_60['feature'] == 'time']['importance_percentage']
        if len(time_d1) > 0 and len(time_azt) > 0:
            latex_lines.append(f"Time feature shows higher importance in D1namo ({time_d1.iloc[0]:.1f}%) compared to AZT1D ({time_azt.iloc[0]:.1f}%) at 60 minutes")
    
    return ' '.join(latex_lines)

# Generate insights
print("\n" + "="*60)
print("FEATURE IMPORTANCE INSIGHTS")
print("="*60)

insights = analyze_feature_importance_insights(all_results)
for insight in insights:
    print(f"- {insight}")

# Save insights
with open('results/feature_importance_insights.txt', 'w') as f:
    f.write('\n'.join(insights))

# Generate LaTeX insights
latex_insights = generate_latex_insights(all_results)
with open('results/feature_importance_latex.txt', 'w') as f:
    f.write(latex_insights + '\n')

print(f"\nLaTeX insights: {latex_insights}")