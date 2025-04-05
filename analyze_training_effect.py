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

# Load evaluation metrics
df = pd.read_csv('analysis/evaluation_metrics.csv')

# Get unique prediction horizons, approaches, and patients
prediction_horizons = sorted(df['Prediction Horizon'].unique())
approaches = df['Approach'].unique()
patients = sorted(df['Patient'].unique())

# Create a new DataFrame to store training data information
training_df = []

# For each test day in the evaluation metrics, determine the amount of training data
for patient in patients:
    for ph in prediction_horizons:
        # Get the complete dataset
        glucose_data, _ = get_data(str(patient).zfill(3), ph)
        
        # Calculate training data for each test day and hour
        for test_day in df['Day'].unique():
            for test_hour in df['Hour'].unique():
                # Training data is all data before the test day or same day but before test hour
                train_data = glucose_data[(glucose_data['datetime'].dt.day < test_day) | 
                                          ((glucose_data['datetime'].dt.day == test_day) & 
                                           (glucose_data['hour'] < test_hour))]
                
                # Count rows (each row is 5 minutes of data)
                train_size = len(train_data)
                
                # Add to our tracking DataFrame
                training_df.append({
                    'Patient': patient,
                    'Prediction Horizon': ph,
                    'Test Day': test_day,
                    'Test Hour': test_hour,
                    'Training Data Count': train_size
                })

training_df = pd.DataFrame(training_df)

# Merge training data information with evaluation metrics
analysis_df = pd.merge(
    df,
    training_df,
    how='left',
    left_on=['Patient', 'Prediction Horizon', 'Day', 'Hour'],
    right_on=['Patient', 'Prediction Horizon', 'Test Day', 'Test Hour']
)

# Calculate average RMSE by prediction horizon, approach, and training data amount
# Group by bins of training data for better visualization
analysis_df['Training Data Bins'] = pd.cut(
    analysis_df['Training Data Count'],
    bins=[0, 500, 750, 1000, 1250, 1500, 2000],
    labels=['<500', '500-750', '750-1000', '1000-1250', '1250-1500', '1500+']
)

# Create a figure specifically for meal features, 30min and 120min horizons
plt.figure(figsize=(14, 6))

# Convert prediction horizons to minutes for easier interpretation
selected_horizons = [6, 24]  # 6*5=30min, 24*5=120min
horizon_minutes = {6: 30, 24: 120}

# Extract only meal features approach data (using pixtral-large-latest which has meal features)
meal_features_data = analysis_df[analysis_df['Approach'] == 'pixtral-large-latest']


# For each selected prediction horizon
for i, ph in enumerate(selected_horizons):
    ax = plt.subplot(1, 2, i+1)
    
    # Filter data for the current prediction horizon
    horizon_data = meal_features_data[meal_features_data['Prediction Horizon'] == ph]

    # Plot all RMSE points against training data amount, colored by patient
    sns.scatterplot(
        data=horizon_data,
        x='Training Data Count',
        y='RMSE',
        hue='Patient',
        palette='viridis',
        alpha=0.7,
        s=50
    )
    
    # Add regression line to show trend
    sns.regplot(
        data=horizon_data,
        x='Training Data Count',
        y='RMSE',
        scatter=False,
        color='red',
        line_kws={'linewidth': 2}
    )
    
    # Calculate Pearson correlation coefficient
    corr, p_value = np.corrcoef(horizon_data['Training Data Count'], horizon_data['RMSE'])[0,1], 0
    
    # Add correlation text
    plt.text(
        0.05, 0.95,
        f"Pearson correlation: {corr:.3f}",
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.8),
        verticalalignment='top'
    )
    
    plt.title(f" {horizon_minutes[ph]} min prediction horizon")
    plt.xlabel('Training Data Amount (rows)')
    plt.ylabel('RMSE')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/supplementary_data/meal_features_training_effect.png', dpi=300)
plt.savefig('images/supplementary_data/meal_features_training_effect.eps', dpi=300)
plt.close()

# Create additional visualization to show hour-of-day effect
plt.figure(figsize=(15, 8))

# Create a pivot table to show average RMSE by hour and day
hour_day_pivot = meal_features_data.pivot_table(
    values='RMSE', 
    index=['Test Hour'], 
    columns=['Test Day'],
    aggfunc='mean'
)

# Generate a summary CSV file with key findings
# Group data by prediction horizon and patient
summary_data = []
for ph in selected_horizons:
    horizon_data = meal_features_data[meal_features_data['Prediction Horizon'] == ph]
    
    # For each patient
    for patient in horizon_data['Patient'].unique():
        patient_data = horizon_data[horizon_data['Patient'] == patient]
        
        # Calculate min and max training data
        min_train = patient_data['Training Data Count'].min()
        max_train = patient_data['Training Data Count'].max()
        
        # Calculate RMSE for min and max training data
        min_rmse = patient_data[patient_data['Training Data Count'] == min_train]['RMSE'].mean()
        max_rmse = patient_data[patient_data['Training Data Count'] == max_train]['RMSE'].mean()
        
        # Calculate Pearson correlation
        corr = np.corrcoef(patient_data['Training Data Count'], patient_data['RMSE'])[0,1]
        
        # Add to summary
        summary_data.append({
            'Prediction Horizon (minutes)': ph * 5,
            'Patient': patient,
            'Min Training Rows': min_train,
            'Max Training Rows': max_train,
            'Initial RMSE': min_rmse,
            'Final RMSE': max_rmse,
            'Correlation': corr
        })

# Create summary dataframe
summary_df = pd.DataFrame(summary_data)

# Also calculate overall averages per prediction horizon
overall_summary = []
for ph in selected_horizons:
    ph_data = summary_df[summary_df['Prediction Horizon (minutes)'] == ph * 5]
    overall_summary.append({
        'Prediction Horizon (minutes)': ph * 5,
        'Patient': 'Average',
        'Min Training Rows': ph_data['Min Training Rows'].mean(),
        'Max Training Rows': ph_data['Max Training Rows'].mean(),
        'Initial RMSE': ph_data['Initial RMSE'].mean(),
        'Final RMSE': ph_data['Final RMSE'].mean(),
        'Correlation': ph_data['Correlation'].mean()
    })

# Add overall averages to summary
summary_df = pd.concat([summary_df, pd.DataFrame(overall_summary)], ignore_index=True)