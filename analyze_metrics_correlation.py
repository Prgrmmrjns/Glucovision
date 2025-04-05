import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Function to get projected value (needed for get_data)
def get_projected_value(window, prediction_horizon):
    x = np.arange(len(window))
    coeffs = np.polyfit(x, window, deg=3)
    return np.polyval(coeffs, len(window) + prediction_horizon)

# Function to get data for a patient
def get_data(patient, prediction_horizon):
    # Load data
    glucose_data = pd.read_csv(f"diabetes_subset_pictures-glucose-food-insulin/{patient}/glucose.csv")
    insulin_data = pd.read_csv(f"diabetes_subset_pictures-glucose-food-insulin/{patient}/insulin.csv")
    food_data = pd.read_csv(f"food_data/pixtral-large-latest/{patient}.csv")

    # Process glucose data
    glucose_data["datetime"] = pd.to_datetime(glucose_data["date"] + ' ' + glucose_data["time"])
    glucose_data = glucose_data.drop(['type', 'comments', 'date', 'time'], axis=1)
    glucose_data['glucose'] *= 18.0182
    glucose_data['hour'] = glucose_data['datetime'].dt.hour
    glucose_data['time'] = glucose_data['hour'] + glucose_data['datetime'].dt.minute / 60

    # Process insulin data
    insulin_data["datetime"] = pd.to_datetime(insulin_data["date"] + ' ' + insulin_data["time"])
    insulin_data['insulin'] = insulin_data['slow_insulin'] + insulin_data['fast_insulin']
    insulin_data = insulin_data.drop(['slow_insulin', 'fast_insulin', 'comment', 'date', 'time'], axis=1)

    # Process food data
    food_data['datetime'] = pd.to_datetime(food_data['datetime'], format='%Y:%m:%d %H:%M:%S')
    food_data = food_data[['datetime', 'simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers']]

    # Combine data
    combined_data = pd.concat([food_data, insulin_data]).sort_values('datetime').reset_index(drop=True)
    combined_data.fillna(0, inplace=True)

    # Calculate target variables
    glucose_data['glucose_next'] = glucose_data['glucose'] - glucose_data['glucose'].shift(-prediction_horizon)
    glucose_data['glucose_change'] = glucose_data['glucose'] - glucose_data['glucose'].shift(1)
    
    window_size = 6
    glucose_data['glucose_change_projected'] = glucose_data['glucose_change'].rolling(
        window=window_size, min_periods=window_size
    ).apply(lambda window: get_projected_value(window, prediction_horizon))
    
    glucose_data['glucose_projected'] = glucose_data['glucose'].rolling(
        window=window_size, min_periods=window_size
    ).apply(lambda window: get_projected_value(window, prediction_horizon))
    
    glucose_data.dropna(subset=['glucose_next'], inplace=True)
    return glucose_data, combined_data

# Set target prediction horizon
prediction_horizon = 12

# Load evaluation metrics
df = pd.read_csv('analysis/evaluation_metrics.csv')

# Get unique approaches and patients
approaches = df['Approach'].unique()
patients = sorted(df['Patient'].unique())

# Extract only meal features approach data (using pixtral-large-latest which has meal features)
meal_features_data = df[(df['Approach'] == 'pixtral-large-latest') & (df['Prediction Horizon'] == prediction_horizon)]

# Create a new DataFrame to store metrics information
metrics_df = []

# For each test day/hour in the evaluation metrics, calculate various metrics
for patient in patients:
    # Get the complete dataset
    glucose_data, combined_data = get_data(str(patient).zfill(3), prediction_horizon)
    
    # Loop through each test day and hour in the evaluation metrics
    for idx, row in meal_features_data[meal_features_data['Patient'] == patient].iterrows():
        test_day = row['Day']
        test_hour = row['Hour']
        
        # Get data for specific day
        day_glucose = glucose_data[glucose_data['datetime'].dt.day == test_day]
        
        # Calculate glucose metrics for the day up to the test hour
        glucose_prior = day_glucose[day_glucose['hour'] < test_hour]
        
        if len(glucose_prior) > 0:
            glucose_mean = glucose_prior['glucose'].mean()
            glucose_std = glucose_prior['glucose'].std()
            glucose_min = glucose_prior['glucose'].min()
            glucose_max = glucose_prior['glucose'].max()
            glucose_range = glucose_max - glucose_min
            glucose_cv = glucose_std / glucose_mean if glucose_mean > 0 else 0
            
            # Calculate rate metrics
            glucose_prior_copy = glucose_prior.copy()
            glucose_prior_copy.loc[:, 'glucose_rate'] = glucose_prior['glucose'].diff() / 5  # Rate per minute
            glucose_rate_mean = glucose_prior_copy['glucose_rate'].mean()
            glucose_rate_std = glucose_prior_copy['glucose_rate'].std()
            glucose_rate_max = glucose_prior_copy['glucose_rate'].max()
            glucose_rate_min = glucose_prior_copy['glucose_rate'].min()
            
            # Get extreme values
            high_glucose_pct = (glucose_prior['glucose'] > 180).mean() * 100  # Percent time above 180 mg/dL
            low_glucose_pct = (glucose_prior['glucose'] < 70).mean() * 100   # Percent time below 70 mg/dL
            
            # Calculate time in range
            time_in_range = ((glucose_prior['glucose'] >= 70) & (glucose_prior['glucose'] <= 180)).mean() * 100
        else:
            # Default values if no prior data
            glucose_mean = glucose_std = glucose_min = glucose_max = glucose_range = glucose_cv = 0
            glucose_rate_mean = glucose_rate_std = glucose_rate_max = glucose_rate_min = 0
            high_glucose_pct = low_glucose_pct = time_in_range = 0
        
        # Get food and insulin data for the day
        day_start = pd.Timestamp(day_glucose['datetime'].dt.date.iloc[0])
        day_end = pd.Timestamp(day_glucose['datetime'].dt.date.iloc[0]) + pd.Timedelta(days=1)
        
        # Filter test day data up to the test hour
        test_hour_time = pd.Timestamp(day_glucose['datetime'].dt.date.iloc[0]) + pd.Timedelta(hours=test_hour)
        
        day_combined = combined_data[(combined_data['datetime'] >= day_start) & 
                                    (combined_data['datetime'] < test_hour_time)]
        
        # Calculate food metrics
        simple_sugars_total = day_combined['simple_sugars'].sum()
        complex_sugars_total = day_combined['complex_sugars'].sum()
        proteins_total = day_combined['proteins'].sum()
        fats_total = day_combined['fats'].sum()
        dietary_fibers_total = day_combined['dietary_fibers'].sum()
        
        # Calculate meal count and size metrics
        meal_threshold = 5  # Consider intake > 5g as a meal
        meals = day_combined[(day_combined['simple_sugars'] > meal_threshold) | 
                             (day_combined['complex_sugars'] > meal_threshold)]
        meal_count = len(meals)
        
        if meal_count > 0:
            meal_size_mean = (meals['simple_sugars'] + meals['complex_sugars']).mean()
            meal_size_max = (meals['simple_sugars'] + meals['complex_sugars']).max()
        else:
            meal_size_mean = meal_size_max = 0
            
        # Calculate insulin metrics
        insulin_total = day_combined['insulin'].sum()
        insulin_max = day_combined['insulin'].max()
        insulin_count = (day_combined['insulin'] > 0).sum()
        
        # Calculate carb-to-insulin ratio
        total_carbs = simple_sugars_total + complex_sugars_total
        carb_insulin_ratio = total_carbs / insulin_total if insulin_total > 0 else 0
        
        # Add all metrics to our DataFrame
        metrics_df.append({
            'Patient': patient,
            'Day': test_day,
            'Hour': test_hour,
            'RMSE': row['RMSE'],
            'glucose_mean': glucose_mean,
            'glucose_std': glucose_std,
            'glucose_min': glucose_min,
            'glucose_max': glucose_max,
            'glucose_range': glucose_range,
            'glucose_cv': glucose_cv,
            'glucose_rate_mean': glucose_rate_mean,
            'glucose_rate_std': glucose_rate_std,
            'glucose_rate_max': glucose_rate_max,
            'glucose_rate_min': glucose_rate_min,
            'high_glucose_pct': high_glucose_pct,
            'low_glucose_pct': low_glucose_pct,
            'time_in_range': time_in_range,
            'simple_sugars_total': simple_sugars_total,
            'complex_sugars_total': complex_sugars_total,
            'proteins_total': proteins_total,
            'fats_total': fats_total,
            'dietary_fibers_total': dietary_fibers_total,
            'insulin_total': insulin_total,
            'insulin_max': insulin_max,
            'insulin_count': insulin_count,
            'meal_count': meal_count,
            'meal_size_mean': meal_size_mean,
            'meal_size_max': meal_size_max,
            'carb_insulin_ratio': carb_insulin_ratio
        })

# Create DataFrame with all metrics
metrics_df = pd.DataFrame(metrics_df)

# Create output directories if they don't exist
os.makedirs('analysis/metrics_correlation', exist_ok=True)
os.makedirs('images/metrics_correlation', exist_ok=True)

# Calculate correlations between metrics and RMSE
correlation_results = []
for col in metrics_df.columns:
    if col not in ['Patient', 'Day', 'Hour', 'RMSE']:
        # Calculate overall correlation
        corr = np.corrcoef(metrics_df[col], metrics_df['RMSE'])[0, 1]
        correlation_results.append({
            'Metric': col,
            'Correlation': corr,
            'Abs_Correlation': abs(corr)
        })

# Create correlation DataFrame
correlation_df = pd.DataFrame(correlation_results)
correlation_df.sort_values('Abs_Correlation', ascending=False, inplace=True)

# Calculate patient-specific correlations
patient_correlations = {}
for patient in patients:
    patient_data = metrics_df[metrics_df['Patient'] == patient]
    patient_corrs = []
    
    for col in patient_data.columns:
        if col not in ['Patient', 'Day', 'Hour', 'RMSE']:
            # Skip calculation if the column has no variance
            if patient_data[col].std() == 0:
                corr = 0
            else:
                corr = np.corrcoef(patient_data[col], patient_data['RMSE'])[0, 1]
            
            patient_corrs.append({
                'Metric': col,
                'Correlation': corr,
                'Abs_Correlation': abs(corr)
            })
    
    patient_correlations[patient] = pd.DataFrame(patient_corrs)
    patient_correlations[patient].sort_values('Abs_Correlation', ascending=False, inplace=True)

# Create bar plot of top 15 metrics by absolute correlation
plt.figure(figsize=(14, 6))
top_metrics = correlation_df.head(15)

# Create bar plot with explicit hue parameter and viridis palette
ax = sns.barplot(
    data=top_metrics,
    x='Metric',
    y='Correlation',
    legend=False  # Don't show legend since it would be redundant
)

# Add correlation values at y=0.2 for each bar
for i, v in enumerate(top_metrics['Correlation']):
    ax.text(i, 0.025, f'{v:.2f}', ha='center', va='center', fontsize=10, fontweight='bold')

plt.xticks(rotation=15, ha='center')
plt.ylabel('Pearson Correlation Coefficient')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

# Save figure
plt.savefig('images/supplementary_data/top_metrics_correlation.png', dpi=300)
plt.savefig('images/supplementary_data/top_metrics_correlation.eps', dpi=300)
plt.close()

# Also create a heatmap of correlations between all metrics and RMSE
# First, compute correlation matrix
correlation_matrix = metrics_df.corr()['RMSE'].sort_values(ascending=False)

# Save all results to CSV
correlation_df.to_csv('analysis/overall_metric_correlations.csv', index=False)