import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from params import *
from processing_functions import *

# Load Bézier parameters
with open(f'{RESULTS_PATH}/d1namo_bezier_params.json', 'r') as f:
    global_params = json.load(f)

# Load data from patient 001
patient = '001'
glucose_data, combined_data = get_d1namo_data(patient)

# Add macronutrient features using Bézier curves
patient_data = add_d1namo_features(global_params, OPTIMIZATION_FEATURES_D1NAMO, (glucose_data, combined_data))

# Create figure with custom width ratios (1:2 for 1/3:2/3)
fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 2], hspace=0.3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

# Colors and names for consistency
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
feature_names = ['Simple Sugars', 'Complex Sugars', 'Proteins', 'Fats', 'Dietary Fibers', 'Insulin']

# Subplot A: Bézier Curves (1/3 width)
ax1.text(-0.13, 0.95, 'A', transform=ax1.transAxes, fontsize=16, va='top', fontweight='bold')

max_x = 0
for i, (feature, params) in enumerate(global_params.items()):
    points = np.array(params).reshape(-1, 2)
    curve = bezier_curve(points)
    max_x = max(max_x, curve[:, 0].max())
    
    ax1.plot(curve[:, 0], curve[:, 1], color=colors[i], linewidth=3, 
             label=feature_names[i], alpha=0.8)

ax1.set_xlabel('Time After Consumption (hours)', fontsize=12)
ax1.set_ylabel('Influence Weight', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.1, max_x + 0.3)
ax1.set_ylim(-0.05, 1.05)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Subplot B: Feature modeling over time (2/3 width)
ax2.text(-0.08, 0.95, 'B', transform=ax2.transAxes, fontsize=16, va='top', fontweight='bold')

# Select a representative time period (first 3 days)
start_time = patient_data['datetime'].min()
end_time = start_time + pd.Timedelta(days=3)
plot_data = patient_data[(patient_data['datetime'] >= start_time) & (patient_data['datetime'] <= end_time)].copy()

# Plot glucose on primary y-axis
line_glucose = ax2.plot(plot_data['datetime'], plot_data['glucose'], 'k-', linewidth=3, 
                       label='Glucose', alpha=0.9)
ax2.set_ylabel('Glucose (mg/dL)', fontsize=12)
ax2.set_xlabel('Time', fontsize=12)
ax2.grid(True, alpha=0.3)

# Create secondary y-axis for macronutrient features
ax2_twin = ax2.twinx()

lines_features = []
for i, feature in enumerate(OPTIMIZATION_FEATURES_D1NAMO):
    if feature in plot_data.columns:
        line = ax2_twin.plot(plot_data['datetime'], plot_data[feature], color=colors[i], 
                           linewidth=2, label=feature_names[i], alpha=0.7)
        lines_features.extend(line)

ax2_twin.set_ylabel('Feature Values (Modeled Influence)', fontsize=12)

# Add food/insulin event markers
for idx, row in combined_data.iterrows():
    if start_time <= row['datetime'] <= end_time:
        has_food = any(row[f] > 0 for f in ['simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers'])
        has_insulin = row['insulin'] > 0
        
        if has_food:
            ax2.axvline(x=row['datetime'], color='red', linestyle='--', alpha=0.5, linewidth=1)
        if has_insulin:
            ax2.axvline(x=row['datetime'], color='blue', linestyle=':', alpha=0.7, linewidth=1)

# Create shared legend at bottom center
legend_elements = ([Line2D([0], [0], color=colors[i], linewidth=2, label=feature_names[i]) for i in range(len(feature_names))] +
                  [Line2D([0], [0], color='black', linewidth=3, label='Glucose')] +
                  [Line2D([0], [0], color='red', linestyle='--', alpha=0.5, label='Food Events')] +
                  [Line2D([0], [0], color='blue', linestyle=':', alpha=0.7, label='Insulin Events')])

fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
           ncol=4, fontsize=10, frameon=False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Make room for legend
plt.savefig('../manuscript/images/results/combined_metabolic_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Subplot A: Bézier curves for {len(global_params)} macronutrient/insulin features")
print(f"Subplot B: {len(plot_data)} glucose measurements over {(end_time-start_time).days} days for Patient {patient}")

# Extract peak times for analysis
peak_times = {}
for feature, params in global_params.items():
    points = np.array(params).reshape(-1, 2)
    curve = bezier_curve(points)
    peak_idx = np.argmax(curve[:, 1])
    peak_times[feature] = curve[peak_idx, 0]

print("\nPeak times for each feature:")
for feature, peak_time in peak_times.items():
    print(f"{feature}: {peak_time:.2f} hours")
