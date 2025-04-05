import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import seaborn as sns
from scipy.special import comb
import os
from matplotlib.gridspec import GridSpec

# Create output directories
os.makedirs('analysis', exist_ok=True)
os.makedirs('images/results', exist_ok=True)

# Load patient parameters
with open('parameters/patient_bezier_params.json', 'r') as f:
    patient_params = json.load(f)

features = ['simple_sugars', 'complex_sugars', 'fats', 'dietary_fibers', 'proteins', 'insulin']
patients = list(patient_params.keys())

# Function to generate Bezier curve from control points
def bezier_curve(points, num=100):
    n = len(points) - 1
    t = np.linspace(0, 1, num)
    curve = np.zeros((num, 2))
    
    for i, point in enumerate(points):
        curve += np.outer(comb(n, i) * (t**i) * ((1-t)**(n-i)), point)
    
    return curve[np.argsort(curve[:, 0])]

# Calculate metrics for each patient
patient_metrics = {patient: {
    'peak_times': [],
    'effect_durations': [],
    'peak_intensities': [],
    'auc': []
} for patient in patients}

# Process all curves
all_curves = {}
max_time = 15  # Maximum time to show on x-axis

for feature in features:
    all_curves[feature] = {}
    for patient in patients:
        params = patient_params[patient][feature]
        control_points = np.array(params).reshape(-1, 2)
        curve = bezier_curve(control_points, num=200)
        all_curves[feature][patient] = curve
        
        # Find peak
        peak_idx = np.argmax(curve[:, 1])
        patient_metrics[patient]['peak_times'].append(curve[peak_idx, 0])
        patient_metrics[patient]['peak_intensities'].append(curve[peak_idx, 1])
        
        # Calculate effect duration (time until effect drops below 10% of peak)
        threshold = 0.1 * curve[peak_idx, 1]
        post_peak = curve[peak_idx:]
        duration_idx = np.where(post_peak[:, 1] < threshold)[0]
        if len(duration_idx) > 0:
            duration = post_peak[duration_idx[0], 0] - curve[peak_idx, 0]
        else:
            duration = curve[-1, 0] - curve[peak_idx, 0]
        patient_metrics[patient]['effect_durations'].append(duration)
        
        # Calculate area under curve (AUC)
        # Limit to max_time for consistent comparison
        mask = curve[:, 0] <= max_time
        limited_curve = curve[mask]
        auc = np.trapz(limited_curve[:, 1], limited_curve[:, 0])
        patient_metrics[patient]['auc'].append(auc)

# Compute summary statistics for each patient
patient_summary = pd.DataFrame({
    'Patient': patients,
    'Avg Peak Time (h)': [np.mean(patient_metrics[p]['peak_times']) for p in patients],
    'Std Peak Time': [np.std(patient_metrics[p]['peak_times']) for p in patients],
    'Avg Effect Duration (h)': [np.mean(patient_metrics[p]['effect_durations']) for p in patients],
    'Std Effect Duration': [np.std(patient_metrics[p]['effect_durations']) for p in patients],
    'Avg Peak Intensity': [np.mean(patient_metrics[p]['peak_intensities']) for p in patients],
    'Std Peak Intensity': [np.std(patient_metrics[p]['peak_intensities']) for p in patients],
    'Avg AUC': [np.mean(patient_metrics[p]['auc']) for p in patients],
    'Total AUC': [np.sum(patient_metrics[p]['auc']) for p in patients]
})

# Sort by average peak time to indicate metabolism speed
patient_summary = patient_summary.sort_values('Avg Peak Time (h)')

# Print patient ranking
print("\n----- PATIENT METABOLISM SPEED ANALYSIS -----\n")
print("Patient ranking by average peak time (faster to slower metabolism):")
for i, (_, row) in enumerate(patient_summary.iterrows()):
    patient = row['Patient']
    print(f"{i+1}. Patient {patient}: Avg Peak Time = {row['Avg Peak Time (h)']:.2f}h, Avg Duration = {row['Avg Effect Duration (h)']:.2f}h")

fastest_patient = patient_summary.iloc[0]['Patient']
slowest_patient = patient_summary.iloc[-1]['Patient']

print(f"\nFastest metabolism: Patient {fastest_patient} - Avg Peak Time: {patient_summary.iloc[0]['Avg Peak Time (h)']:.2f}h")
print(f"Slowest metabolism: Patient {slowest_patient} - Avg Peak Time: {patient_summary.iloc[-1]['Avg Peak Time (h)']:.2f}h")

# Create feature-specific metrics for each patient
detailed_metrics = []
for patient in patients:
    for i, feature in enumerate(features):
        detailed_metrics.append({
            'Patient': patient,
            'Feature': feature,
            'Peak Time (h)': patient_metrics[patient]['peak_times'][i],
            'Effect Duration (h)': patient_metrics[patient]['effect_durations'][i],
            'Peak Intensity': patient_metrics[patient]['peak_intensities'][i],
            'AUC': patient_metrics[patient]['auc'][i]
        })

detailed_df = pd.DataFrame(detailed_metrics)

# Calculate average curves for each feature
avg_curves = {}
std_curves = {}

for feature in features:
    # Interpolate all curves to the same x values for averaging
    x_values = np.linspace(0, max_time, 200)
    interpolated_y = np.zeros((len(patients), len(x_values)))
    
    for i, patient in enumerate(patients):
        curve = all_curves[feature][patient]
        # Only use x values up to max_time
        mask = curve[:, 0] <= max_time
        # Interpolate
        interpolated_y[i] = np.interp(x_values, curve[mask, 0], curve[mask, 1], right=0)
    
    avg_curve = np.column_stack((x_values, np.mean(interpolated_y, axis=0)))
    std_curve = np.column_stack((x_values, np.std(interpolated_y, axis=0)))
    
    avg_curves[feature] = avg_curve
    std_curves[feature] = std_curve

# Create a figure with a 2x3 grid layout
fig = plt.figure(figsize=(24, 12))
gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 1])

# 1. Average curves plot (top row, spans all columns)
ax1 = fig.add_subplot(gs[0, :])

for i, feature in enumerate(features):
    avg_curve = avg_curves[feature]
    std_curve = std_curves[feature]
    color = plt.cm.tab10(i)
    ax1.plot(avg_curve[:, 0], avg_curve[:, 1], label=feature, linewidth=2, color=color)
    ax1.fill_between(
        avg_curve[:, 0], 
        avg_curve[:, 1] - std_curve[:, 1], 
        avg_curve[:, 1] + std_curve[:, 1], 
        alpha=0.2, color=color
    )

# Add bold 'A' label to upper left corner
ax1.text(-0.05, 1.05, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold')

ax1.set_xlabel('Time (hours)', fontsize=12)
ax1.set_ylabel('Effect Strength', fontsize=12)
ax1.set_title('Average Effect Curves Across Patients', fontsize=14)
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, max_time)
ax1.set_ylim(0, 1.0)

# 2. Radar chart for Peak Times (bottom row, first column)
ax2 = fig.add_subplot(gs[1, 0], polar=True)

# Add bold 'B' label to upper left corner
ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold')

# Set radar chart angles
angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
angles += angles[:1]  # Close the loop

# Extended feature list for closing the loop
radar_features = features + [features[0]]

# Create a colormap for patients
patient_colors = plt.cm.viridis(np.linspace(0, 1, len(patients)))

# Plot each patient on radar chart for peak times
for i, patient in enumerate(patient_summary['Patient']):
    # Get peak times for this patient
    values = [detailed_df[(detailed_df['Patient'] == patient) & (detailed_df['Feature'] == f)]['Peak Time (h)'].values[0] 
             for f in features]
    
    # Normalize by the maximum peak time across all patients and features
    max_peak_time = detailed_df['Peak Time (h)'].max()
    values_norm = [v/max_peak_time for v in values]
    
    # Close the loop
    values_norm += [values_norm[0]]
    
    # Plot
    ax2.plot(angles, values_norm, 'o-', linewidth=2, label=f'Patient {patient}', color=patient_colors[i])
    ax2.fill(angles, values_norm, alpha=0.1, color=patient_colors[i])

ax2.set_thetagrids(np.degrees(angles)[:-1], radar_features[:-1])
ax2.set_title('Peak Effect Times', fontsize=14)

# 3. Radar chart for Peak Intensities (bottom row, second column)
ax3 = fig.add_subplot(gs[1, 1], polar=True)

# Add bold 'C' label to upper left corner
ax3.text(-0.1, 1.1, 'C', transform=ax3.transAxes, fontsize=16, fontweight='bold')

# Plot each patient on radar chart for peak intensities
for i, patient in enumerate(patient_summary['Patient']):
    # Get peak intensities for this patient
    values = [detailed_df[(detailed_df['Patient'] == patient) & (detailed_df['Feature'] == f)]['Peak Intensity'].values[0] 
             for f in features]
    
    # Normalize by the maximum peak intensity across all patients and features
    max_peak_intensity = detailed_df['Peak Intensity'].max()
    values_norm = [v/max_peak_intensity for v in values]
    
    # Close the loop
    values_norm += [values_norm[0]]
    
    # Plot
    ax3.plot(angles, values_norm, 'o-', linewidth=2, color=patient_colors[i])
    ax3.fill(angles, values_norm, alpha=0.1, color=patient_colors[i])

ax3.set_thetagrids(np.degrees(angles)[:-1], radar_features[:-1])
ax3.set_title('Peak Effect Intensities', fontsize=14)

# 4. Radar chart for Effect Durations (bottom row, third column)
ax4 = fig.add_subplot(gs[1, 2], polar=True)

# Add bold 'D' label to upper left corner
ax4.text(-0.1, 1.1, 'D', transform=ax4.transAxes, fontsize=16, fontweight='bold')

# Plot each patient on radar chart for effect durations
for i, patient in enumerate(patient_summary['Patient']):
    # Get effect durations for this patient
    values = [detailed_df[(detailed_df['Patient'] == patient) & (detailed_df['Feature'] == f)]['Effect Duration (h)'].values[0] 
             for f in features]
    
    # Normalize by the maximum effect duration across all patients and features
    max_effect_duration = detailed_df['Effect Duration (h)'].max()
    values_norm = [v/max_effect_duration for v in values]
    
    # Close the loop
    values_norm += [values_norm[0]]
    
    # Plot
    ax4.plot(angles, values_norm, 'o-', linewidth=2, color=patient_colors[i])
    ax4.fill(angles, values_norm, alpha=0.1, color=patient_colors[i])

ax4.set_thetagrids(np.degrees(angles)[:-1], radar_features[:-1])
ax4.set_title('Effect Durations', fontsize=14)

# Add a shared legend for all radar charts
handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(patients), bbox_to_anchor=(0.5, 0), fontsize=10)

# Create a colorbar to indicate metabolism ranking
cbar_ax = fig.add_axes([0.89, 0.05, 0.02, 0.4])
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=len(patients)-1))
sm.set_array([])
cbar = plt.colorbar(sm, cax=cbar_ax)
cbar.set_ticks([0, len(patients)-1])
cbar.set_ticklabels(['Faster', 'Slower'])
cbar.set_label('Relative Metabolism Speed', rotation=270, labelpad=15)

plt.tight_layout(rect=[0, 0.05, 0.9, 1])
plt.savefig('images/results/metabolism_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Save detailed metrics to CSV
detailed_df.to_csv('analysis/patient_detailed_metrics.csv', index=False)
patient_summary.to_csv('analysis/patient_metabolism_summary.csv', index=False)

print("\nAnalysis complete. Results saved to 'images/results/metabolism_analysis.png'")
print("Detailed metrics available in CSV files in the 'analysis' directory.") 