import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

# Load the data
df = pd.read_csv("data/6_008.csv")

# Define features to sum
features = ['simple_sugars', 'complex_sugars', 'fats', 'dietary_fibers', 'proteins', 'insulin']

# Convert datetime
df['datetime'] = pd.to_datetime(df['datetime'])

# Limit to the last 300 entries for visualization consistency
df_limited = df[-300:-100]

# Calculate the sum of all features
df_limited['feature_sum'] = df_limited[features].sum(axis=1)

# Initialize the scaler
scaler = MinMaxScaler()

# Apply scaling to glucose_next and feature_sum
# Use .values.reshape(-1, 1) for single columns
df_limited['glucose_next_scaled'] = scaler.fit_transform(df_limited[['glucose_next']])
df_limited['feature_sum_scaled'] = scaler.fit_transform(df_limited[['feature_sum']])


# Create a figure
plt.figure(figsize=(12, 4))

# Plot scaled glucose_next and feature_sum
plt.plot(df_limited['datetime'], df_limited['glucose_next_scaled'], 'b-', linewidth=3, label='Glucose')
plt.plot(df_limited['datetime'], df_limited['feature_sum_scaled'], 'r-', linewidth=3, alpha=0.7, label='Feature Sum')

# Minimalistic styling
plt.xticks([])
plt.yticks([])
plt.gca().xaxis_date() # Format x-axis for dates, although ticks are hidden

# Remove frame/borders
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

# Add legend
plt.legend(loc='upper right', fontsize=14)

# Ensure layout is tight
plt.tight_layout()

# Save the plot
plt.savefig('graphical_abstract/features_vs_glucose_next.png')
plt.close()

# Create a figure
plt.figure(figsize=(12, 4))

# Plot scaled glucose_next and feature_sum
plt.plot(df_limited['datetime'], df_limited['glucose_next_scaled'], 'b-', linewidth=3, label='Glucose')

# Minimalistic styling
plt.xticks([])
plt.yticks([])
plt.gca().xaxis_date() # Format x-axis for dates, although ticks are hidden

# Remove frame/borders
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

# Add legend
plt.legend(loc='upper right', fontsize=14)

# Ensure layout is tight
plt.tight_layout()

# Save the plot
plt.savefig('graphical_abstract/glucose_next.png')
plt.close()


predictions_df = pd.read_csv("predictions/pixtral-large-latest/6/008_predictions.csv")

# Convert datetime
predictions_df['Datetime'] = pd.to_datetime(predictions_df['Datetime'])

# Filter predictions to match the time range in df_limited
min_date = df_limited['datetime'].min()
max_date = df_limited['datetime'].max()
predictions_filtered = predictions_df[
    (predictions_df['Datetime'] >= min_date) & 
    (predictions_df['Datetime'] <= max_date)
]

# Create a new scaler for predictions and ground truth
pred_scaler = MinMaxScaler()

# Scale both columns
predictions_filtered['Predictions_scaled'] = pred_scaler.fit_transform(predictions_filtered[['Predictions']])
predictions_filtered['Ground_truth_scaled'] = pred_scaler.fit_transform(predictions_filtered[['Ground_truth']])

# Create a figure for predictions vs ground truth
plt.figure(figsize=(16, 4))

# Plot predictions and ground truth
plt.plot(predictions_filtered['Datetime'], predictions_filtered['Ground_truth_scaled'], 'b-', linewidth=3, label='Glucose')
plt.plot(predictions_filtered['Datetime'], predictions_filtered['Predictions_scaled'], 'g-', linewidth=3, alpha=0.7, label='Predictions')

# Minimalistic styling
plt.xticks([])
plt.yticks([])
plt.gca().xaxis_date()

# Remove frame/borders
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

# Add legend
plt.legend(loc='upper right', fontsize=14)

# Ensure layout is tight
plt.tight_layout()

# Save the plot
plt.savefig('graphical_abstract/predictions_vs_ground_truth.png')
plt.close()
