import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

# --- Bezier Curve Functions ---

def bezier_curve(points, num=100):
    """Generate Bezier curve from control points using Bernstein polynomials"""
    n = len(points) - 1
    if n != 3:
        raise ValueError("This function requires 4 control points for a cubic Bezier curve.")
    t = np.linspace(0, 1, num)
    curve = np.zeros((num, 2))
    for i, point in enumerate(points):
        # Calculate Bernstein polynomial basis and add contribution
        curve += np.outer(comb(n, i) * (t**i) * ((1-t)**(n-i)), np.array(point))
    # Ensure curve is sorted by x-values for interpolation/lookup
    # Note: The Bernstein polynomial generation naturally follows t from 0 to 1.
    # If P1.x > P2.x etc., the x-values might not be monotonic initially.
    # Sorting ensures lookup works correctly.
    curve = curve[np.argsort(curve[:, 0])]
    return curve

def get_impact(curve, delta_t):
    """Find the impact (y-value) on the curve for a given elapsed time (delta_t)"""
    x_curve, y_curve = curve[:, 0], curve[:, 1]
    max_time = x_curve[-1] # P3's x-coordinate (end time)

    # Check if delta_t is within the curve's time range [0, max_time]
    if delta_t < 0 or delta_t > max_time + 1e-9: # Add tolerance
         return 0.0

    # Find the index of the closest x-value on the curve
    # Using interpolation might be slightly more accurate than nearest point lookup
    # but for visualization, nearest point is often sufficient.
    idx = np.abs(x_curve - delta_t).argmin()

    # Handle edge case where delta_t might be slightly beyond the last calculated x
    if idx == len(x_curve) - 1 and delta_t > x_curve[idx]:
        # If delta_t is truly beyond the curve end time (P3.x), return 0
        # This check is slightly redundant with the initial check but safe
         if delta_t > max_time + 1e-9:
              return 0.0
         else: # Otherwise, use the last point's y-value (should be close to 0)
              return y_curve[idx]

    # Basic nearest-neighbor lookup
    return y_curve[idx]

    # Alternative: Linear Interpolation (requires curve points to be sorted by x)
    # return np.interp(delta_t, x_curve, y_curve)


# --- Simulation Parameters ---

# Define the cubic control points for the simple sugars example
# P0=(0,0), P1=(0.25, 1.0), P2=(0.7, -0.2), P3=(1.8, 0.0)  <- P2.y is now negative
simple_sugar_points_cubic = np.array([[0.0, 0.0], [0.5, 1.0], [1, 0], [3, 0.0]])

# Generate the cubic curve points
curve_cubic = bezier_curve(simple_sugar_points_cubic, num=500) # Higher resolution for plotting

# Define meal times (hours relative to 11:00 AM) and amounts
meal1_time_rel = 0.0 # 11:00 AM
meal2_time_rel = 1.0 # 12:00 PM
meal1_simple_sugars = 25.0 # grams
meal2_simple_sugars = 20.0 # grams

# Define specific times of interest for calculation/annotation
time_1200_rel = 1.0 # 12:00 PM
time_1230_rel = 1.5 # 12:30 PM

# --- Calculate Impacts at Specific Times ---

delta_t1_1200 = time_1200_rel - meal1_time_rel # 1.0
delta_t2_1200 = time_1200_rel - meal2_time_rel # 0.0
impact1_1200_factor = get_impact(curve_cubic, delta_t1_1200)
impact2_1200_factor = get_impact(curve_cubic, delta_t2_1200)
total_modeled_impact_1200 = (meal1_simple_sugars * impact1_1200_factor) + (meal2_simple_sugars * impact2_1200_factor)

delta_t1_1230 = time_1230_rel - meal1_time_rel # 1.5
delta_t2_1230 = time_1230_rel - meal2_time_rel # 0.5
impact1_1230_factor = get_impact(curve_cubic, delta_t1_1230)
impact2_1230_factor = get_impact(curve_cubic, delta_t2_1230)
total_modeled_impact_1230 = (meal1_simple_sugars * impact1_1230_factor) + (meal2_simple_sugars * impact2_1230_factor)

# --- Calculate Impacts Over Time for Plotting ---

# Time axis for plotting (e.g., 11:00 AM to 2:30 PM)
time_axis_hours = np.linspace(0, 3.5, 1000) # Increased resolution for smoother curves

effective_simple_sugars1_over_time = np.zeros_like(time_axis_hours)
effective_simple_sugars2_over_time = np.zeros_like(time_axis_hours)

for i, t in enumerate(time_axis_hours):
    # Elapsed time since each meal
    delta_t1 = t - meal1_time_rel
    delta_t2 = t - meal2_time_rel

    # Get impact factor for each meal at time t
    impact1_factor = get_impact(curve_cubic, delta_t1)
    impact2_factor = get_impact(curve_cubic, delta_t2)

    # Calculate effective simple sugars from each meal using new variable names
    effective_simple_sugars1_over_time[i] = meal1_simple_sugars * impact1_factor
    effective_simple_sugars2_over_time[i] = meal2_simple_sugars * impact2_factor

# Calculate total modeled simple sugar impact over time using new variable names
total_modeled_impact_over_time = effective_simple_sugars1_over_time + effective_simple_sugars2_over_time

# --- Plotting ---

plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                               gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.05})

# Create a secondary y-axis for the Bezier curve values
ax1_twin = ax1.twinx()

# Plot effective amounts on ax1 with colors to match axes
# Keep only the total sugar impact (left axis)
ax1.plot(time_axis_hours, total_modeled_impact_over_time, color='blue', linewidth=2)

# Plot the actual Bezier curve on the secondary y-axis with distinct styling
ax1_twin.plot(curve_cubic[:, 0], curve_cubic[:, 1], color='darkred', linewidth=1.5, 
             linestyle='-', alpha=0.5)

# Create Bezier value curves for individual meals (on right axis)
meal1_impact_factors = np.zeros_like(time_axis_hours)
meal2_impact_factors = np.zeros_like(time_axis_hours)

for i, t in enumerate(time_axis_hours):
    # Elapsed time since each meal
    delta_t1 = t - meal1_time_rel
    delta_t2 = t - meal2_time_rel

    # Get impact factor for each meal at time t
    meal1_impact_factors[i] = get_impact(curve_cubic, delta_t1)
    meal2_impact_factors[i] = get_impact(curve_cubic, delta_t2)

# Add Meal 1 Bezier curve values to right axis - using dashed line
ax1_twin.plot(time_axis_hours, meal1_impact_factors, color='darkred', linestyle='--', 
             alpha=0.8, linewidth=1.5, label='Meal 1 Bezier Curve')

# Add Meal 2 Bezier curve values to right axis
ax1_twin.plot(time_axis_hours, meal2_impact_factors, color='darkred', linestyle=':', 
             alpha=0.8, linewidth=1.5, label='Meal 2 Bezier Curve')

ax1_twin.set_ylim(0, 1)
ax1_twin.set_ylabel('Bezier Curve y-value', color='darkred')
ax1_twin.tick_params(axis='y', labelcolor='darkred')

# Add annotations for specific Bezier y-values
annotation_fontsize = 8
# At 12:00 (t=1.0)
ax1_twin.annotate(f'M1: {impact1_1200_factor:.2f}', 
                  xy=(time_1200_rel, impact1_1200_factor), 
                  xytext=(time_1200_rel, impact1_1200_factor + 0.05), 
                  ha='center', fontsize=annotation_fontsize, color='darkred',
                  arrowprops=dict(arrowstyle='-', color='darkred', linestyle='--', alpha=0.7))
ax1_twin.annotate(f'M2: {impact2_1200_factor:.2f}', 
                  xy=(time_1200_rel, impact2_1200_factor), 
                  xytext=(time_1200_rel - 0.05, impact2_1200_factor + 0.05), # Shift left and up
                  ha='right', # Align text right
                  va='bottom', # Align text bottom
                  fontsize=annotation_fontsize, color='darkred',
                  arrowprops=dict(arrowstyle='-', color='darkred', linestyle=':', alpha=0.7))
# At 12:30 (t=1.5)
ax1_twin.annotate(f'M1: {impact1_1230_factor:.2f}', 
                  xy=(time_1230_rel, impact1_1230_factor), 
                  xytext=(time_1230_rel + 0.1, impact1_1230_factor + 0.1), 
                  ha='left', fontsize=annotation_fontsize, color='darkred',
                  arrowprops=dict(arrowstyle='-', color='darkred', linestyle='--', alpha=0.7))
ax1_twin.annotate(f'M2: {impact2_1230_factor:.2f}', 
                  xy=(time_1230_rel, impact2_1230_factor), 
                  xytext=(time_1230_rel + 0.1, impact2_1230_factor - 0.05), 
                  ha='left', va='top', fontsize=annotation_fontsize, color='darkred',
                  arrowprops=dict(arrowstyle='-', color='darkred', linestyle=':', alpha=0.7))

# Annotate ax1 at specific times using new variable names and phrasing
ax1.annotate(f'Total: {total_modeled_impact_1200:.1f}g',
             xy=(time_1200_rel, total_modeled_impact_1200),
             xytext=(time_1200_rel, total_modeled_impact_1200 - 2),
             ha='center', arrowprops=dict(arrowstyle='->', color='blue'))
ax1.annotate(f'Total: {total_modeled_impact_1230:.1f}g',
             xy=(time_1230_rel, total_modeled_impact_1230),
             xytext=(time_1230_rel, total_modeled_impact_1230 - 2),
             ha='center', arrowprops=dict(arrowstyle='->', color='blue'))

# Update y-axis label and color
ax1.set_ylabel('Cumulative Modeled Simple Sugars Impact (g)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Combine legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1_twin.legend(lines + lines2, labels + labels2, loc='upper right')

ax1.grid(True, linestyle='--', alpha=0.7)


# Plot meal events on ax2 (Labels remain the same)
# Call vlines separately for each meal to set different linestyles
ax2.vlines(meal1_time_rel, ymin=0, ymax=1, color='darkred', linestyle='--', lw=2,
           label='Meal 1 Intake') # Use Meal 1 linestyle
ax2.vlines(meal2_time_rel, ymin=0, ymax=1, color='darkred', linestyle=':', lw=2,
           label='Meal 2 Intake') # Use Meal 2 linestyle
ax2.text(meal1_time_rel, 0.5, f' Meal 1\n {meal1_simple_sugars:.0f}g Simple Sugars', ha='left', va='center', color='darkred')
ax2.text(meal2_time_rel, 0.5, f' Meal 2\n {meal2_simple_sugars:.0f}g Simple Sugars', ha='left', va='center', color='darkred')

ax2.set_yticks([]) # No y-ticks needed for event plot
ax2.set_ylim(0, 1)
ax2.grid(True, axis='x', linestyle='--', alpha=0.7)

# Set x-axis ticks to represent clock times (optional but helpful)
hour_ticks = np.arange(0, 4.0, 0.5)
hour_labels = [f"{11+int(h):02d}:{int((h % 1) * 60):02d}" for h in hour_ticks]
ax2.set_xticks(hour_ticks)
ax2.set_xticklabels(hour_labels)
ax1.set_xlim(left=-0.1, right=3.6) # Adjust limits slightly

plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout to prevent title overlap
plt.savefig('images/methods/example_bezier.png') # Save the plot
plt.savefig('images/methods/example_bezier.eps') # Save the plot
plt.close()

print("--- Calculated Values ---")
print(f"Impact Factor @ 12:00 (Meal 1, \u0394t=1.0h): {impact1_1200_factor:.3f}")
print(f"Impact Factor @ 12:00 (Meal 2, \u0394t=0.0h): {impact2_1200_factor:.3f}")
print(f"Total Modeled Simple Sugar Impact @ 12:00 (t=1.0h): {total_modeled_impact_1200:.2f}g")
print(f"Impact Factor @ 12:30 (Meal 1, \u0394t=1.5h): {impact1_1230_factor:.3f}")
print(f"Impact Factor @ 12:30 (Meal 2, \u0394t=0.5h): {impact2_1230_factor:.3f}")
print(f"Total Modeled Simple Sugar Impact @ 12:30 (t=1.5h): {total_modeled_impact_1230:.2f}g")
print("\nPlot saved as cubic_bezier_impact_visualization.png")