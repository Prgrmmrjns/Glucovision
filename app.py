import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image
import os
import pickle
from scipy.special import comb
import lightgbm as lgb

# Load Bezier parameters
with open('parameters/patient_bezier_params.json', 'r') as f:
    patient_params = json.load(f)

# Constants
patients = ['001', '002', '004', '006', '007', '008']
prediction_horizons = [6, 9, 12, 18, 24]
meal_features = ['simple_sugars', 'complex_sugars', 'fats', 'dietary_fibers', 'proteins']
features = meal_features + ['insulin']

# Function to generate Bezier curve
def bezier_curve(points, num=50):
    n = len(points) - 1
    t = np.linspace(0, 1, num)
    curve = np.zeros((num, 2))
    
    for i, point in enumerate(points):
        curve += np.outer(comb(n, i) * (t**i) * ((1-t)**(n-i)), point)
    
    return curve[np.argsort(curve[:, 0])]

# Function to load meal data (filtered to last two days)
def load_meal_data(patient):
    df = pd.read_csv(f"food_data/pixtral-large-latest/{patient}.csv")
    # Add insulin column if it doesn't exist
    if 'insulin' not in df.columns:
        df['insulin'] = 0.0
    
    # Convert datetime
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y:%m:%d %H:%M:%S')
    
    # Filter to last two days only
    unique_days = df['datetime'].dt.date.unique()
    if len(unique_days) > 2:
        last_two_days = sorted(unique_days)[-2:]
        df = df[df['datetime'].dt.date.isin(last_two_days)]
        
    return df

# Function to load image
def load_image(patient, image_name):
    img_path = f"diabetes_subset_pictures-glucose-food-insulin/{patient}/food_pictures/{image_name}"
    try:
        return Image.open(img_path)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

# Function to display feature importances
def display_feature_importances(importances):
    if not importances:
        return None
        
    # Sort by importance value
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    features = [item[0] for item in sorted_importances]
    values = [item[1] for item in sorted_importances]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(features, values, color='skyblue')
    
    # Add percentage labels inside bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', va='center')
    
    ax.set_xlabel('Importance (%)')
    ax.set_title('Feature Importances')
    plt.tight_layout()
    return fig

# Function for feature preprocessing using Bezier curves
def process_features(meal_data, patient, prediction_horizon):
    # Extract features and current datetime
    feature_values = {}
    for feature in features:
        feature_values[feature] = meal_data[feature].iloc[0] if feature in meal_data.columns else 0.0
    
    # Parse datetime (no need here as we only need the values and patient params)
    
    # Process features using Bezier curves - calculate CHANGE in impact
    processed_features_change = {}
    for feature in features:
        # Access points correctly from the nested structure
        params = np.array(patient_params[patient]['bezier_points'][feature]).reshape(-1, 2)
        curve = bezier_curve(params, num=100)
        x_curve, y_curve = curve[:, 0], curve[:, 1]
        
        # Calculate impact change based on Bezier curve
        # Impact at horizon - Impact at time 0
        time_point_horizon = prediction_horizon / 12  # Convert prediction interval to hours
        time_point_start = 0.0

        # Interpolate to find effect strength at start and horizon
        impact_factor_horizon = np.interp(time_point_horizon, x_curve, y_curve, left=0.0, right=0.0) # Assume 0 effect outside curve range
        impact_factor_start = np.interp(time_point_start, x_curve, y_curve, left=0.0, right=0.0) # Should be 0 if curve starts at (0,0)
        
        # Calculate the change in impact scaled by the feature value
        impact_change = (impact_factor_horizon - impact_factor_start)
        processed_features_change[feature] = feature_values[feature] * impact_change
    
    # Return the CHANGE in processed features and the original raw features
    return processed_features_change, feature_values

# Function to modify macronutrient values and process features
def modify_and_process_features(meal_data, patient, prediction_horizon, modifications):
    # Make a copy of the meal data
    modified_meal_data = meal_data.copy()
    
    # Apply modifications
    for feature, change in modifications.items():
        if feature in modified_meal_data.columns:
            modified_meal_data[feature] = max(0, modified_meal_data[feature].iloc[0] + change)
    
    # Process with modified values using the updated process_features function
    return process_features(modified_meal_data, patient, prediction_horizon)

# Function to visualize feature impact
def prepare_feature_visualization(patient):
    fig, ax = plt.subplots(figsize=(10, 6))
    for feature in features:
        # Access points correctly from the nested structure
        params = np.array(patient_params[patient]['bezier_points'][feature]).reshape(-1, 2)
        curve = bezier_curve(params, num=100)
        ax.plot(curve[:, 0], curve[:, 1], label=feature, linewidth=2)
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Effect Strength')
    ax.set_title(f'Bezier Curves for Patient {patient}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

# Function to get projected value using 3rd degree polynomial (from main.py)
def get_projected_value(window, prediction_horizon):
    # Ensure window is a pandas Series for iloc access
    if not isinstance(window, pd.Series):
        window = pd.Series(window)
        
    if len(window) < 4: # Need at least 4 points for 3rd degree polyfit
        if len(window) > 0:
           return window.iloc[-1] # Return last value if not enough points
        else:
           return 0.0 # Or return 0 if window is empty

    x = np.arange(len(window))
    # Use numpy's polyfit and polyval
    coeffs = np.polyfit(x, window.values, deg=3)
    # Project `prediction_horizon` steps into the future
    return np.polyval(coeffs, len(window) -1 + prediction_horizon)

# UI
st.title("Glucose Prediction App")

# Layout with columns
col1, col2 = st.columns([1, 2])

# Sidebar for selection
with col1:
    st.header("Parameters")
    selected_patient = st.selectbox("Select Patient", patients)
    
    # Load meal data for selected patient
    meal_data = load_meal_data(selected_patient)
    
    # Get image options
    image_options = meal_data['picture'].tolist()
    selected_image = st.selectbox("Select Image", image_options)
    
    # Prediction horizon
    selected_horizon = st.selectbox("Select Prediction Horizon (in 5-min intervals)", prediction_horizons)
    
    # Macronutrient modification section
    st.header("Modify Macronutrients")
    st.write("Adjust macronutrient values to see effect on glucose prediction:")
    
    # Initialize modification values
    modifications = {}
    for feature in meal_features:
        modifications[feature] = st.slider(
            f"Adjust {feature}", 
            min_value=-20, 
            max_value=20, 
            value=0,
            step=1,
            help=f"Change the amount of {feature}"
        )
    
    # Button for prediction
    predict_button = st.button("Submit")

# Main display area
with col2:
    # Display selected image
    st.header("Selected Meal")
    meal_row = meal_data[meal_data['picture'] == selected_image]
    if not meal_row.empty:
        image = load_image(selected_patient, selected_image)
        if image:
            st.image(image, caption=selected_image)
        
        # Display meal data
        st.header("Meal Data")
        st.write(f"**Description:** {meal_row['description'].iloc[0]}")
        st.write(f"**Date and Time:** {meal_row['datetime'].iloc[0]}")
            
        # Create nutritional values table
        nutrition_df = pd.DataFrame({
            'Nutrient': meal_features,
            'Original Value': [meal_row[f].iloc[0] for f in meal_features],
            'Modified Value': [max(0, meal_row[f].iloc[0] + modifications.get(f, 0)) for f in meal_features] # Use .get for safety
        })
        st.table(nutrition_df)
    else:
        st.warning("Selected image not found in recent meal data.")

# Prediction section
if predict_button:
    st.header("Prediction")
    meal_row = meal_data[meal_data['picture'] == selected_image]
    
    if not meal_row.empty:
        
        # --- Load Pre-trained Model ---
        model_path = f'models/pixtral-large-latest/{selected_horizon}/patient_{selected_patient}_model.pkl'
        model = None
        model_feature_names = None
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                model_feature_names = model.feature_name_ # Get feature names from model
            except Exception as e:
                st.error(f"Error loading model: {e}")
        else:
            st.error(f"Pre-trained model not found for patient {selected_patient}, horizon {selected_horizon}.")
            st.stop() # Stop execution if model not found
            
        # --- End Load Model ---
        
        # Create two columns for results display
        left_col, right_col = st.columns(2)
        
        with left_col:
            # Display feature impact curves (Bezier curves)
            st.subheader("Feature Impact Curves")
            fig = prepare_feature_visualization(selected_patient)
            st.pyplot(fig)
            
            # Display feature importances from loaded model
            st.subheader("Feature Importances")
            importances = dict(zip(model_feature_names, (model.feature_importances_ / model.feature_importances_.sum()) * 100))
            importance_fig = display_feature_importances(importances)
            if importance_fig:
                st.pyplot(importance_fig)
            else:
                st.write("Could not display feature importances.")

        # --- Feature Processing and Vector Creation ---
        # Process original features (get CHANGE in impact)
        original_processed_features_change, original_features = process_features(meal_row, selected_patient, selected_horizon)
        
        # Process modified features (get CHANGE in impact)
        modified_processed_features_change, modified_features = modify_and_process_features(
            meal_row, selected_patient, selected_horizon, modifications
        )
        
        # Get model feature names from the loaded model (already done above)
        
        # Create feature vectors with all expected columns
        original_feature_vector = pd.Series(index=model_feature_names, dtype=float).fillna(0.0)
        modified_feature_vector = pd.Series(index=model_feature_names, dtype=float).fillna(0.0)
        
        # Load real glucose data for the patient (needed for glucose context)
        glucose_data = pd.read_csv(f"diabetes_subset_pictures-glucose-food-insulin/{selected_patient}/glucose.csv")
        glucose_data["datetime"] = pd.to_datetime(glucose_data["date"] + ' ' + glucose_data["time"])
        glucose_data = glucose_data.sort_values('datetime').drop(['type', 'comments', 'date', 'time'], axis=1)
        glucose_data['glucose'] *= 18.0182  # Convert to mg/dL
        
        # Parse meal datetime
        dt_str = meal_row['datetime'].iloc[0]
        try:
            meal_dt = pd.to_datetime(dt_str, format='%Y:%m:%d %H:%M:%S')
        except:
            meal_dt = pd.to_datetime(dt_str)
            
        # Get glucose readings before or at the meal time
        glucose_before_meal = glucose_data[glucose_data['datetime'] <= meal_dt].copy() # Use .copy()

        # Ensure there is glucose data before the meal
        if glucose_before_meal.empty:
            st.error(f"No historical glucose data found before the selected meal time for patient {selected_patient}.")
            st.stop()

        current_glucose = glucose_before_meal['glucose'].iloc[-1]
        
        # Calculate glucose change (handle case with only one reading)
        if len(glucose_before_meal) > 1:
            glucose_change = current_glucose - glucose_before_meal['glucose'].iloc[-2]
        else:
            glucose_change = 0.0 # Assume no change if only one prior point

        # Calculate projected glucose values using the updated function
        window_size = 6 # As used in main.py for projection input
        glucose_window = glucose_before_meal['glucose'].iloc[-window_size:]
        
        # Calculate projected change (use diff for changes)
        glucose_changes = glucose_window.diff().dropna() # Get series of changes
        glucose_change_projected = get_projected_value(glucose_changes, selected_horizon)
        
        # Calculate projected glucose level
        glucose_projected = get_projected_value(glucose_window, selected_horizon)
        
        # Populate the feature vectors
        for vector, processed_change in zip([original_feature_vector, modified_feature_vector],
                                             [original_processed_features_change, modified_processed_features_change]):
            
            # Set values for processed feature CHANGES
            for feature in features:
                if feature in vector.index: # Check if feature is in model features
                    vector[feature] = processed_change[feature]
            
            # Add real-time/contextual features
            if 'time' in vector.index:
                vector['time'] = meal_dt.hour + meal_dt.minute / 60
            if 'glucose' in vector.index:
                vector['glucose'] = current_glucose
            if 'glucose_change' in vector.index:
                vector['glucose_change'] = glucose_change
            if 'glucose_change_projected' in vector.index:
                vector['glucose_change_projected'] = glucose_change_projected
            if 'glucose_projected' in vector.index:
                vector['glucose_projected'] = glucose_projected
            
            # Fill any remaining NaNs (features not calculated above) with 0
            vector.fillna(0.0, inplace=True)

        # Make predictions using the loaded model
        # Ensure the order matches model_feature_names by reindexing
        original_feature_array = original_feature_vector[model_feature_names].values.reshape(1, -1)
        modified_feature_array = modified_feature_vector[model_feature_names].values.reshape(1, -1)
        
        try:
            original_prediction = model.predict(original_feature_array)[0]
            modified_prediction = model.predict(modified_feature_array)[0]
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.stop()

        # --- End Feature Processing and Vector Creation ---

        with right_col:
            # Display original prediction
            st.subheader(f"Predicted Glucose Change (in {selected_horizon*5} minutes)")
            if original_prediction > 0:
                st.error(f"↑ +{original_prediction:.1f} mg/dL (increase)")
            else:
                st.success(f"↓ {original_prediction:.1f} mg/dL (decrease)")
            
            # Display modified prediction
            st.subheader(f"Predicted Modified Glucose Change (in {selected_horizon*5} minutes)")
            if modified_prediction > 0:
                st.error(f"↑ +{modified_prediction:.1f} mg/dL (increase)")
            else:
                st.success(f"↓ {modified_prediction:.1f} mg/dL (decrease)")
            
            # Display the difference
            difference = modified_prediction - original_prediction
            st.subheader("Effect of Modifications")
            if difference > 0:
                st.error(f"↑ +{difference:.1f} mg/dL higher than original")
            elif difference < 0:
                st.success(f"↓ {abs(difference):.1f} mg/dL lower than original")
            else:
                st.info("No change from modifications")
    else:
        # Handle case where meal_row was empty after button press (should be rare)
        st.warning("Selected image data not found.")
                
# Add info about the app
st.sidebar.markdown("---")
st.sidebar.info("""
This app uses Bezier curves to model how different nutrients affect glucose levels over time.
The models are trained on historical data and patient-specific parameters.
Data are based on the D1namo dataset (https://www.sciencedirect.com/science/article/pii/S2352914818301059) where
Type 1 Diabetes patients were monitored for around 5 days with CGM and asked to upload meal images and insulin data.
Select a patient, meal image, and prediction horizon, then click Submit to see the predicted glucose change.

You can also modify macronutrient values to see how changing your meal composition affects glucose prediction.
Check out https://github.com/Prgrmmrjns/Glucovision for more information.
""") 