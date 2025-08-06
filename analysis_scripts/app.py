import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image
import os
import pickle
import lightgbm as lgb
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from params import *
from processing_functions import *

# Load Bezier parameters
@st.cache_data
def load_bezier_params():
    with open(f'{RESULTS_PATH}/d1namo_bezier_params.json', 'r') as f:
        return json.load(f)

# Function to load meal data (filtered to last two days)
@st.cache_data
def load_meal_data(patient):
    df = pd.read_csv(f"{FOOD_DATA_PATH}/{patient}.csv")
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

# Function to load glucose data for a patient
@st.cache_data
def load_glucose_data(patient):
    glucose_data = pd.read_csv(f"{D1NAMO_DATA_PATH}/{patient}/glucose.csv")
    glucose_data["datetime"] = pd.to_datetime(glucose_data["date"] + ' ' + glucose_data["time"])
    glucose_data = glucose_data.sort_values('datetime').drop(['type', 'comments', 'date', 'time'], axis=1)
    glucose_data['glucose'] *= 18.0182  # Convert to mg/dL
    glucose_data['hour'] = glucose_data['datetime'].dt.hour
    glucose_data['time'] = glucose_data['hour'] + glucose_data['datetime'].dt.minute / 60
    return glucose_data

# Function to load image
def load_image(patient, image_name):
    img_path = f"{D1NAMO_DATA_PATH}/{patient}/food_pictures/{image_name}"
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

# Function to visualize Bézier curves
def prepare_feature_visualization(patient):
    global_params = load_bezier_params()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    feature_names = ['Simple Sugars', 'Complex Sugars', 'Proteins', 'Fats', 'Dietary Fibers', 'Insulin']
    
    for i, feature in enumerate(OPTIMIZATION_FEATURES_D1NAMO):
        # Get Bézier curve
        curve = bezier_curve(np.array(global_params[feature]).reshape(-1, 2), num=100)
        ax.plot(curve[:, 0], curve[:, 1], label=feature_names[i], linewidth=2, color=colors[i])
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Effect Strength')
    ax.set_title(f'Bézier Curves for All Features')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

# Function to train and predict using d1namo methodology
def make_prediction(patient, meal_datetime, modifications, prediction_horizon):
    global_params = load_bezier_params()
    
    # Load all patient data
    all_data_list = []
    for p in PATIENTS_D1NAMO:
        glucose_data, combined_data = get_d1namo_data(p)
        
        # Apply modifications to the target patient's meal data
        if p == patient:
            # Find the closest meal time and apply modifications
            meal_dt = pd.to_datetime(meal_datetime)
            time_diff = np.abs(combined_data['datetime'] - meal_dt)
            closest_meal_idx = time_diff.idxmin()
            
            # Apply modifications
            for feature, change in modifications.items():
                if feature in combined_data.columns:
                    combined_data.loc[closest_meal_idx, feature] = max(0, 
                        combined_data.loc[closest_meal_idx, feature] + change)
        
        # Process features
        patient_data = add_d1namo_features(global_params, OPTIMIZATION_FEATURES_D1NAMO, 
                                         (glucose_data, combined_data))
        patient_data['patient_id'] = f"patient_{p}"
        all_data_list.append(patient_data)
    
    all_data = pd.concat(all_data_list, ignore_index=True)
    
    # Get test point close to meal time
    patient_mask = all_data['patient_id'] == f"patient_{patient}"
    meal_dt = pd.to_datetime(meal_datetime)
    
    # Find data point closest to meal time
    patient_data = all_data[patient_mask].copy()
    time_diff = np.abs(patient_data['datetime'] - meal_dt)
    closest_idx = time_diff.idxmin()
    test_point = patient_data.loc[[closest_idx]]
    
    if test_point.empty:
        return None, None, "No data found near meal time"
    
    # Prepare training data (all data before test time + other patients)
    X = pd.concat([
        all_data[patient_mask & (all_data['datetime'] < test_point['datetime'].iloc[0])], 
        all_data[~patient_mask]
    ])
    
    if len(X) < 100:
        return None, None, "Not enough training data"
    
    # Features and target
    target_feature = f'glucose_{prediction_horizon}'
    features_to_remove_ph = FEATURES_TO_REMOVE_D1NAMO + [f'glucose_{h}' for h in PREDICTION_HORIZONS]
    available_features = X.columns.difference(features_to_remove_ph)
    
    # Train-validation split with patient weighting
    from sklearn.model_selection import train_test_split
    indices = train_test_split(range(len(X)), test_size=0.2, random_state=42)
    current_patient_weight = 10
    weights = [np.where(X['patient_id'].values[idx] == f"patient_{patient}", current_patient_weight, 1) 
               for idx in indices]
    
    train = X[available_features]
    X_train, y_train, weights_train = train.values[indices[0]], X[target_feature].values[indices[0]], weights[0]
    X_val, y_val, weights_val = train.values[indices[1]], X[target_feature].values[indices[1]], weights[1]
    
    # Train model
    model = lgb.train(
        LGB_PARAMS, 
        lgb.Dataset(X_train, label=y_train, weight=weights_train),
        valid_sets=[lgb.Dataset(X_val, label=y_val, weight=weights_val)]
    )
    
    # Make prediction
    test_features = test_point[available_features]
    prediction = model.predict(test_features.values)[0]
    current_glucose = test_point['glucose'].iloc[0]
    
    # Get feature importances
    feature_names = list(available_features)
    feature_importance_values = model.feature_importance(importance_type='gain')
    importances = dict(zip(feature_names, (feature_importance_values / feature_importance_values.sum()) * 100))
    
    return prediction, current_glucose, importances

# UI
st.title("Glucovision")
st.subheader("An innovative approach for leveraging meal images for glucose forecasting and patient metabolic modeling")

# Layout with columns
col1, col2 = st.columns([1, 2])

# Sidebar for selection
with col1:
    st.header("Parameters")
    selected_patient = st.selectbox("Select Patient", PATIENTS_D1NAMO)
    
    # Load meal data for selected patient
    meal_data = load_meal_data(selected_patient)
    
    # Get image options
    image_options = meal_data['picture'].tolist()
    selected_image = st.selectbox("Select Image", image_options)
    
    # Prediction horizon
    horizon_options = {6: "30 min", 12: "60 min", 18: "90 min", 24: "120 min"}
    selected_horizon = st.selectbox("Select Prediction Horizon", 
                                   options=list(horizon_options.keys()),
                                   format_func=lambda x: horizon_options[x])
    
    # Macronutrient modification section
    st.header("Modify Macronutrients")
    st.write("Adjust macronutrient values to see effect on glucose prediction:")
    
    # Initialize modification values
    modifications = {}
    meal_features = ['simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers']
    for feature in meal_features:
        modifications[feature] = st.slider(
            f"Adjust {feature.replace('_', ' ').title()}", 
            min_value=-20, 
            max_value=20, 
            value=0,
            step=1,
            help=f"Change the amount of {feature.replace('_', ' ')}"
        )
    
    # Button for prediction
    predict_button = st.button("Predict Glucose Change", type="primary")

# Main display area
with col2:
    if selected_image:
        # Display selected image
        st.header("Selected Meal")
        meal_row = meal_data[meal_data['picture'] == selected_image]
        if not meal_row.empty:
            image = load_image(selected_patient, selected_image)
            if image:
                st.image(image, caption=selected_image, width=300)
            
            # Display meal data
            st.header("Meal Data")
            st.subheader("The following macronutrient values were estimated by a multimodal Large Language Model")
            st.write(f"**Description:** {meal_row['description'].iloc[0] if 'description' in meal_row.columns else 'N/A'}")
            st.write(f"**Date and Time:** {meal_row['datetime'].iloc[0]}")
                
            # Create nutritional values table
            nutrition_df = pd.DataFrame({
                'Nutrient': [f.replace('_', ' ').title() for f in meal_features],
                'Original Value (g)': [meal_row[f].iloc[0] if f in meal_row.columns else 0 for f in meal_features],
                'Modified Value (g)': [max(0, (meal_row[f].iloc[0] if f in meal_row.columns else 0) + modifications.get(f, 0)) for f in meal_features]
            })
            st.table(nutrition_df)
        else:
            st.warning("Selected image not found in meal data.")
    else:
        st.info("Please select a meal image to continue.")

# Prediction section
if predict_button and selected_image:
    st.header("Prediction Results")
    meal_row = meal_data[meal_data['picture'] == selected_image]
    
    if not meal_row.empty:
        with st.spinner("Training model and making prediction..."):
            prediction, current_glucose, importances = make_prediction(
                selected_patient, 
                meal_row['datetime'].iloc[0], 
                modifications, 
                selected_horizon
            )
        
        if prediction is not None:
            # Create two columns for results
            left_col, right_col = st.columns(2)
            
            with left_col:
                # Display Bézier curves
                st.subheader("Feature Impact Curves")
                fig = prepare_feature_visualization(selected_patient)
                st.pyplot(fig)
            
            with right_col:
                # Display prediction results
                st.subheader(f"Predicted Glucose Change")
                st.metric(
                    label=f"In {horizon_options[selected_horizon]}",
                    value=f"{prediction:+.1f} mg/dL",
                    delta=f"From current: {current_glucose:.1f} mg/dL"
                )
                
                if prediction > 0:
                    st.error(f"⚠️ Expected glucose increase of {prediction:.1f} mg/dL")
                else:
                    st.success(f"✅ Expected glucose decrease of {abs(prediction):.1f} mg/dL")
            
            # Display feature importances
            st.subheader("Model Feature Importances")
            if importances and isinstance(importances, dict):
                importance_fig = display_feature_importances(importances)
                if importance_fig:
                    st.pyplot(importance_fig)
                else:
                    st.write("Could not display feature importances.")
            else:
                st.write("Feature importances not available.")
                
        else:
            st.error("Could not make prediction. Please try a different meal or patient.")
    else:
        st.warning("Selected meal data not found.")

# Add info about the app
st.sidebar.markdown("---")
st.sidebar.info("""
**Enhanced D1namo Glucose Prediction**

This app uses:
- Optimized Bézier curves for temporal modeling
- Cross-patient learning with patient weighting
- Real-time feature importance analysis
- LightGBM machine learning

Based on the D1namo dataset with Type 1 Diabetes patients monitored for ~5 days with CGM, meal images, and insulin data.

**How to use:**
1. Select a patient and meal image
2. Choose prediction horizon (30-120 minutes)
3. Optionally modify macronutrients
4. Click "Predict" to see expected glucose change

Check out https://github.com/Prgrmmrjns/Glucovision for more information.
""")

# Display technical details in expander
with st.expander("Technical Details"):
    st.write(f"""
    **Model Configuration:**
    - Patients: {len(PATIENTS_D1NAMO)}
    - Features: {len(OPTIMIZATION_FEATURES_D1NAMO)} optimized Bézier curves
    - ML Algorithm: LightGBM with patient weighting (10:1)
    - Cross-validation: 80/20 train/validation split
    - Prediction horizons: 30, 60, 90, 120 minutes
    
    **Bézier Curve Features:**
    - Simple sugars, complex sugars, proteins, fats, dietary fibers, insulin
    - Globally optimized parameters across all patients
    - Temporal modeling of nutrient absorption and insulin action
    """)