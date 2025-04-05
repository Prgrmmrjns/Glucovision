import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image
import os
import pickle
from scipy.special import comb

# Load Bezier parameters
with open('parameters/patient_bezier_params.json', 'r') as f:
    patient_params = json.load(f)

# Constants
patients = ['001', '002', '004', '006', '007', '008']
prediction_horizons = [6, 9, 12, 18, 24]
meal_features = ['simple_sugars', 'complex_sugars', 'fats', 'dietary_fibers', 'proteins']
features = meal_features + ['insulin']
features_to_remove = ['glucose_next', 'datetime', 'hour']

# Function to generate Bezier curve
def bezier_curve(points, num=50):
    n = len(points) - 1
    t = np.linspace(0, 1, num)
    curve = np.zeros((num, 2))
    
    for i, point in enumerate(points):
        curve += np.outer(comb(n, i) * (t**i) * ((1-t)**(n-i)), point)
    
    return curve[np.argsort(curve[:, 0])]

# Function to load meal data
def load_meal_data(patient):
    df = pd.read_csv(f"food_data/pixtral-large-latest/{patient}.csv")
    # Add insulin column if it doesn't exist
    if 'insulin' not in df.columns:
        df['insulin'] = 0.0
    return df

# Function to load sample data to get feature names
def get_model_feature_names(patient, prediction_horizon):
    # Try to load the sample data that was used to train the model
    sample_data_path = f"data/{prediction_horizon}_{patient}.csv"
    if os.path.exists(sample_data_path):
        df = pd.read_csv(sample_data_path)
        return [col for col in df.columns if col not in features_to_remove]
    else:
        return None

# Function to load image
def load_image(patient, image_name):
    img_path = f"diabetes_subset_pictures-glucose-food-insulin/{patient}/food_pictures/{image_name}"
    try:
        return Image.open(img_path)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

# Function to get feature importances for a model
def get_feature_importances(patient, prediction_horizon):
    model_path = f'models/pixtral-large-latest/{prediction_horizon}/patient_{patient}_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            importances = model.feature_importances_
            feature_names = model.feature_name_
            # Convert to relative importance (percentages)
            relative_importances = (importances / importances.sum()) * 100
            return dict(zip(feature_names, relative_importances))
    return {}

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
    
    # Parse datetime with explicit format
    dt_str = meal_data['datetime'].iloc[0]
    try:
        meal_datetime = pd.to_datetime(dt_str, format='%Y:%m:%d %H:%M:%S')
    except:
        # If the first format fails, try another common format
        meal_datetime = pd.to_datetime(dt_str)
    
    # Process features using Bezier curves
    processed_features = {}
    for feature in features:
        # Convert points to numpy array for processing
        params = np.array(patient_params[patient][feature]).reshape(-1, 2)
        curve = bezier_curve(params, num=100)
        x_curve, y_curve = curve[:, 0], curve[:, 1]
        
        # Calculate impact based on Bezier curve
        # For a single meal, we get the impact at prediction_horizon time
        time_point = prediction_horizon/12  # Convert to hours
        impact_factor = np.interp(time_point, x_curve, y_curve)
        processed_features[feature] = feature_values[feature] * impact_factor
    
    return processed_features, feature_values

# Function to modify macronutrient values and process features
def modify_and_process_features(meal_data, patient, prediction_horizon, modifications):
    # Make a copy of the meal data
    modified_meal_data = meal_data.copy()
    
    # Apply modifications
    for feature, change in modifications.items():
        if feature in modified_meal_data.columns:
            modified_meal_data[feature] = max(0, modified_meal_data[feature].iloc[0] + change)
    
    # Process with modified values
    return process_features(modified_meal_data, patient, prediction_horizon)

# Function to visualize feature impact
def prepare_feature_visualization(patient):
    fig, ax = plt.subplots(figsize=(10, 6))
    for feature in features:
        params = np.array(patient_params[patient][feature]).reshape(-1, 2)
        curve = bezier_curve(params, num=100)
        ax.plot(curve[:, 0], curve[:, 1], label=feature, linewidth=2)
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Effect Strength')
    ax.set_title(f'Bezier Curves for Patient {patient}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

# Function to get projected value from a window of values
def get_projected_value(window, prediction_horizon):
    if len(window) < 2:
        return window.iloc[-1]
    
    # Simple linear projection based on recent trend
    recent_change = window.iloc[-1] - window.iloc[0]
    change_per_interval = recent_change / (len(window) - 1)
    projected_change = change_per_interval * prediction_horizon
    return window.iloc[-1] + projected_change

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
    image = load_image(selected_patient, selected_image)
    st.image(image, caption=selected_image, use_container_width=True)
    
    # Display meal data
    st.header("Meal Data")
    meal_row = meal_data[meal_data['picture'] == selected_image]
    st.write(f"**Description:** {meal_row['description'].iloc[0]}")
    st.write(f"**Date and Time:** {meal_row['datetime'].iloc[0]}")
        
    # Create nutritional values table
    nutrition_df = pd.DataFrame({
        'Nutrient': meal_features,
        'Original Value': [meal_row[f].iloc[0] for f in meal_features],
        'Modified Value': [max(0, meal_row[f].iloc[0] + modifications[f]) for f in meal_features]
    })
    st.table(nutrition_df)

# Prediction section
if predict_button:
    st.header("Prediction")
    meal_row = meal_data[meal_data['picture'] == selected_image]
    
    if not meal_row.empty:
        # Create two columns for results display
        left_col, right_col = st.columns(2)
        
        with left_col:
            # Display feature impact curves (Bezier curves)
            st.subheader("Feature Impact Curves")
            fig = prepare_feature_visualization(selected_patient)
            st.pyplot(fig)
            
            # Check if model exists
            model_path = f'models/pixtral-large-latest/{selected_horizon}/patient_{selected_patient}_model.pkl'
            
            if os.path.exists(model_path):
                try:
                    # Load the model
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    # Display feature importances
                    st.subheader("Feature Importances")
                    importances = get_feature_importances(selected_patient, selected_horizon)
                    importance_fig = display_feature_importances(importances)
                    st.pyplot(importance_fig)
                        
                    # Process original features
                    original_processed_features, original_features = process_features(meal_row, selected_patient, selected_horizon)
                    
                    # Process modified features
                    modified_processed_features, modified_features = modify_and_process_features(
                        meal_row, selected_patient, selected_horizon, modifications
                    )
                    
                    # Get model feature names if available
                    model_feature_names = get_model_feature_names(selected_patient, selected_horizon)
                    
                    model_feature_names = model.feature_name_
                    
                    # Create feature vectors with all expected columns
                    original_feature_vector = {}
                    modified_feature_vector = {}
                    
                    # Set default values for all features
                    for feature_name in model_feature_names:
                        original_feature_vector[feature_name] = 0.0
                        modified_feature_vector[feature_name] = 0.0
                    
                    # Set values for processed features
                    for feature in features:
                        original_feature_vector[feature] = original_processed_features[feature]
                        modified_feature_vector[feature] = modified_processed_features[feature]
                    
                    # Add placeholder values for additional features used by the model
                    for vector in [original_feature_vector, modified_feature_vector]:
                        # Load real glucose data for the patient
                        glucose_data = pd.read_csv(f"diabetes_subset_pictures-glucose-food-insulin/{selected_patient}/glucose.csv")
                        glucose_data["datetime"] = pd.to_datetime(glucose_data["date"] + ' ' + glucose_data["time"])
                        glucose_data = glucose_data.drop(['type', 'comments', 'date', 'time'], axis=1)
                        glucose_data['glucose'] *= 18.0182  # Convert to mg/dL
                        
                        # Parse meal datetime
                        dt_str = meal_row['datetime'].iloc[0]
                        try:
                            meal_dt = pd.to_datetime(dt_str, format='%Y:%m:%d %H:%M:%S')
                        except:
                            meal_dt = pd.to_datetime(dt_str)
                        
                        # Get closest glucose reading before meal
                        glucose_before_meal = glucose_data[glucose_data['datetime'] <= meal_dt]
                        
                        current_glucose = glucose_before_meal.iloc[-1]['glucose']
                        
                        # Calculate time features
                        vector['time'] = meal_dt.hour + meal_dt.minute/60
                        
                        # Set current glucose
                        vector['glucose'] = current_glucose
                        
                        # Calculate glucose change (if we have enough history)
                        vector['glucose_change'] = current_glucose - glucose_before_meal.iloc[-2]['glucose']

                        
                        # Calculate projected glucose values
                        window_size = 6
                        glucose_window = glucose_before_meal.iloc[-window_size:]['glucose']
                        
                        # Calculate projected change
                        glucose_changes = glucose_window.diff().dropna()
                        vector['glucose_change_projected'] = get_projected_value(glucose_changes, selected_horizon)

                        # Calculate projected glucose
                        vector['glucose_projected'] = get_projected_value(glucose_window, selected_horizon)

                    # Make predictions
                    original_feature_array = np.array([original_feature_vector[feature] for feature in model_feature_names])
                    modified_feature_array = np.array([modified_feature_vector[feature] for feature in model_feature_names])
                    
                    original_prediction = model.predict([original_feature_array])[0]
                    modified_prediction = model.predict([modified_feature_array])[0]

                    with right_col:
                        # Display original prediction
                        st.subheader(f"Original Glucose Change (in {selected_horizon*5} minutes)")
                        if original_prediction > 0:
                            st.error(f"↑ +{original_prediction:.1f} mg/dL (increase)")
                        else:
                            st.success(f"↓ {original_prediction:.1f} mg/dL (decrease)")
                        
                        # Display modified prediction
                        st.subheader(f"Modified Glucose Change (in {selected_horizon*5} minutes)")
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
                        
                        # Show processed feature values
                        st.write("**Processed Features for Prediction:**")
                        
                        # Create DataFrame for display
                        processed_data = []
                        for feature in features:
                            original_value = original_features[feature]
                            modified_value = modified_features[feature]
                            
                            # Calculate impact factors (handle division by zero)
                            original_impact = 0 if original_value == 0 else original_processed_features[feature] / original_value
                            modified_impact = 0 if modified_value == 0 else modified_processed_features[feature] / modified_value
                            
                            processed_data.append({
                                'Feature': feature,
                                'Original Value': original_value,
                                'Modified Value': modified_value,
                                'Original Processed': original_processed_features[feature],
                                'Modified Processed': modified_processed_features[feature],
                                'Original Impact': original_impact,
                                'Modified Impact': modified_impact
                            })
                        
                        processed_df = pd.DataFrame(processed_data)
                        st.table(processed_df)
                        
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
                    st.exception(e)  # Show full error details
            else:
                st.error(f"Model not found for patient {selected_patient} with prediction horizon {selected_horizon}")

# Add info about the app
st.sidebar.markdown("---")
st.sidebar.info("""
This app uses Bezier curves to model how different nutrients affect glucose levels over time.
The models are trained on historical data and patient-specific parameters.
Select a patient, meal image, and prediction horizon, then click Submit to see the predicted glucose change.

You can also modify macronutrient values to see how changing your meal composition affects glucose prediction.
Check out https://github.com/Prgrmmrjns/Glucovision for more information.
""") 