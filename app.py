import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from warnings import simplefilter, filterwarnings


# Suppress warnings for cleaner output
filterwarnings("ignore", category=FutureWarning)
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

# Constants
PATIENTS = ['001', '002', '004', '006', '007', '008']

# Prediction horizon in 5-minute intervals (e.g., 6 intervals = 30 minutes)
prediction_horizon = 6

# Features used in the model
features = ['simple_sugars', 'complex_sugars', 'fats', 'dietary_fibers', 'proteins', 'fast_insulin', 'slow_insulin']

# Metabolism parameters for features (sensitivity and peak time)
feature_params = {
    'simple_sugars': [0.5, 0.5],
    'complex_sugars': [0.3, 0.5],
    'proteins': [0.2, 3.5],
    'fats': [0.05, 3.5], 
    'dietary_fibers': [0.03, 3.5],
    'fast_insulin': [1.0, 0.5], 
    'slow_insulin': [0.5, 1.0]
}

def get_trend_intercept(window):
    """
    Calculate the linear trend (intercept) of a rolling window.
    """
    x = np.arange(len(window))
    coeffs = np.polyfit(x, window, deg=2)
    return coeffs[1]  

def get_available_images(patient):
    """
    Get list of available food images for a patient.
    """
    image_dir = f"diabetes_subset_pictures-glucose-food-insulin/{patient}/food_pictures"
    if os.path.exists(image_dir):
        return [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    return []

def add_features_and_create_patient_data(params, features, prediction_horizon):
    """
    Add metabolic features to glucose data based on food and insulin intake.
    """
    glucose_times = glucose_data['datetime'].values.astype('datetime64[s]').astype(np.int64)
    combined_times = combined_data['datetime'].values.astype('datetime64[s]').astype(np.int64)

    for feature in features:
        metabolism_rate, peak_time = params[feature]
        # Calculate time differences in hours
        time_diff_hours = ((glucose_times[:, None] - combined_times[None, :]) / 3600)
        
        # Initialize weights
        weights = np.zeros_like(time_diff_hours)
        
        # Calculate weights based on metabolic model
        increase_mask = (time_diff_hours >= 0) & (time_diff_hours < peak_time)
        weights[increase_mask] = time_diff_hours[increase_mask] / peak_time
        
        plateau_duration = 0.25  # Duration the effect plateaus
        plateau_mask = (time_diff_hours >= peak_time) & (time_diff_hours < peak_time + plateau_duration)
        weights[plateau_mask] = 1
        
        decrease_mask = time_diff_hours >= peak_time + plateau_duration
        weights[decrease_mask] = 1 - ((time_diff_hours[decrease_mask] - peak_time - plateau_duration) * metabolism_rate)
        weights = np.clip(weights, 0, None)
        
        # Apply weights to feature values
        glucose_data[feature] = np.dot(weights, combined_data.loc[:, feature].values)
        glucose_data[feature] = glucose_data[feature] - glucose_data[feature].shift(-prediction_horizon) + glucose_data['glucose_change']

    return glucose_data

def prepare_glucose_data(glucose_data, prediction_horizon, meal_time, selected_food_data, insulin_data):
    # Prepare glucose data
    glucose_data['hour'] = glucose_data['datetime'].dt.hour
    glucose_data['glucose_next'] = glucose_data['glucose'] - glucose_data['glucose'].shift(-prediction_horizon)
    glucose_data['glucose_change'] = glucose_data['glucose'] - glucose_data['glucose'].shift(1)
    glucose_data['glucose_change_6'] = glucose_data['glucose'] - glucose_data['glucose'].shift(6)
    glucose_data['glucose_change_sh_1'] = glucose_data['glucose_change'].shift(1)
    glucose_data['glucose_change_sh_3'] = glucose_data['glucose_change'].shift(3)
    glucose_data['glucose_change_std2'] = glucose_data['glucose_change'].rolling(window=2).std()
    glucose_data['glucose_change_std3'] = glucose_data['glucose_change'].rolling(window=3).std()
    glucose_data['glucose_change_std6'] = glucose_data['glucose_change'].rolling(window=6).std()

    # Calculate RSI
    delta = glucose_data['glucose'].diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=6).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=6).mean()
    glucose_data['glucose_rsi'] = 100 - (100 / (1 + gain / loss))

    # Compute trend intercepts
    glucose_data['glucose_change_trend_intercept'] = glucose_data['glucose_change'].rolling(
        window=6, min_periods=6
    ).apply(get_trend_intercept)
    glucose_data['glucose_trend_intercept'] = glucose_data['glucose'].rolling(
        window=6, min_periods=6
    ).apply(get_trend_intercept)
    glucose_data.dropna(subset=['glucose_next'], inplace=True)

    # Update combined data
    combined_data = pd.concat([selected_food_data, insulin_data]).sort_values('datetime').reset_index(drop=True)
    combined_data.fillna(0, inplace=True)

    # Filter data within 2 hours after meal time
    time_mask = (combined_data['datetime'] >= meal_time) & \
                (combined_data['datetime'] <= meal_time + pd.Timedelta(hours=2))
    combined_data = combined_data[time_mask]
    
    return glucose_data, combined_data

st.markdown("<h3 style='margin-top:-2rem;'>Glucovision: Glucose Level Forecasting Using Multimodal LLMs 👀</h3>", unsafe_allow_html=True)
st.markdown("""
<small>Welcome to Glucovision, a framework designed to forecast glucose levels in Type 1 Diabetes patients by incorporating meal image information using multimodal LLMs.

**Note:** This app is a demonstration using the D1namo dataset and is not intended for medical advice.</small>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Settings")
    selected_patient = st.selectbox(
        "Patient",
        PATIENTS,
        format_func=lambda x: f"Patient {x}"
    )

    available_images = get_available_images(selected_patient)

    if not available_images:
        st.warning(f"No images available for Patient {selected_patient}.")
        st.stop()

    selected_image = st.selectbox(
        "Image",
        available_images,
        format_func=lambda x: x.split('.')[0].replace('_', ' ').title()
    )

    selected_model = st.selectbox(
        "Model",
        ['llama', 'gpt4o', 'sonnet'],
        format_func=lambda x: x.upper()
    )

# Load data
@st.cache_data
def load_data(patient, model):
    # Load food data
    food_data = pd.read_csv(f"food_data/{model}/{patient}.csv")
    # Load insulin data
    insulin_data = pd.read_csv(f"diabetes_subset_pictures-glucose-food-insulin/{patient}/insulin.csv")
    # Load glucose data
    glucose_data = pd.read_csv(f"diabetes_subset_pictures-glucose-food-insulin/{patient}/glucose.csv")
    return food_data, insulin_data, glucose_data

food_data, insulin_data, glucose_data = load_data(selected_patient, selected_model)
# Preprocess data
glucose_data["datetime"] = pd.to_datetime(glucose_data["date"] + ' ' + glucose_data["time"])
glucose_data['glucose'] *= 18.0182  # Convert to mg/dL
glucose_data.drop(['type', 'comments', 'date', 'time'], axis=1, inplace=True)
insulin_data["datetime"] = pd.to_datetime(insulin_data["date"] + ' ' + insulin_data["time"])
insulin_data.drop(['comment', 'date', 'time'], axis=1, inplace=True)
food_data['datetime'] = pd.to_datetime(food_data['datetime'], format='%Y:%m:%d %H:%M:%S')

# Combine food and insulin data
combined_data = pd.concat([food_data, insulin_data]).sort_values('datetime').reset_index(drop=True)
combined_data.fillna(0, inplace=True)

# Extract meal information
selected_food_data = food_data[food_data.index <= food_data[food_data['picture'] == selected_image].index[0]]


# Update meal time (if necessary)
meal_time = selected_food_data['datetime'].iloc[-1]
message = selected_food_data['message'].iloc[-1]

selected_day =  int(meal_time.strftime('%d'))
st.markdown("<h5>Selected meal image</h5>", unsafe_allow_html=True)

image_path = f"diabetes_subset_pictures-glucose-food-insulin/{selected_patient}/food_pictures/{selected_image}"
col1, col2 = st.columns([1, 2])

with col1:
    st.image(image_path, caption=f"Patient {selected_patient} - {selected_image}", width=200)

with col2:
    with st.expander("View LLM Reasoning"):
        st.write(message)
    with st.expander("View LLM Estimations"):
        for feature in features:
            if feature == 'fast_insulin' or feature == 'slow_insulin':
                continue
            st.write(f"{feature}: {selected_food_data[feature].iloc[-1]}")

# Call the function
glucose_data, combined_data = prepare_glucose_data(glucose_data, prediction_horizon, meal_time, selected_food_data, insulin_data)

# Add features
processed_data = add_features_and_create_patient_data(feature_params, features, prediction_horizon)
processed_data.dropna(inplace=True)

# Apply same time filter to processed data
time_mask = (processed_data['datetime'] >= meal_time) & \
            (processed_data['datetime'] <= meal_time + pd.Timedelta(hours=2))
processed_data = processed_data[time_mask]

# Check if there's enough data for prediction
if len(processed_data) < prediction_horizon:
    st.error("⚠️ Not enough data available to make predictions. Please select a different time period or meal.")
    st.stop()

# Load model and prepare data
X_test = processed_data.drop(['datetime', 'glucose_next'], axis=1)
y_test = processed_data['glucose'] - processed_data['glucose_next']
model = joblib.load(f'models/{selected_model}/6/1_{selected_patient}.joblib')

# Train Random Forest for feature importances
model = joblib.load(f'models/{selected_model}/6/{selected_day}_001.joblib')
feature_importances = pd.Series(model.feature_importances_, index=X_test.columns)

# Add tabs for the outputs
st.markdown("<h5>Results</h5>", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["Predictions", "Feature Importances"])

with tab1:
    st.markdown("<h5>Predictions</h5>", unsafe_allow_html=True)

    # Filter predictions starting 30 minutes after meal time
    time_mask_30min = processed_data['datetime'] >= meal_time + pd.Timedelta(minutes=30)
    predictions = model.predict(X_test[time_mask_30min])
    predictions = processed_data[time_mask_30min]['glucose'] - predictions

    # Check for hyper/hypoglycemic predictions
    hyper_mask = predictions > 180
    hypo_mask = predictions < 70

    if any(hyper_mask):
        hyper_times = processed_data.loc[time_mask_30min][hyper_mask]['datetime'].dt.strftime('%H:%M').tolist()
        st.warning(f"⬆️ 🔴 High glucose predicted at: {', '.join(hyper_times)}")
    elif any(hypo_mask):
        hypo_times = processed_data.loc[time_mask_30min][hypo_mask]['datetime'].dt.strftime('%H:%M').tolist() 
        st.warning(f"⬇️ 💙 Low glucose predicted at: {', '.join(hypo_times)}")
    else:
        st.success("✨ 🎯 Glucose levels are predicted to stay in safe range for all timepoints!")

    # Calculate RMSE for predictions 30 minutes onwards
    rmse = np.sqrt(np.mean((predictions - y_test[time_mask_30min]) ** 2))

    # Define glycemic zones
    def glycemic_zone(glucose_level):
        if glucose_level < 70:
            return 'Hypoglycemia'
        elif 70 <= glucose_level <= 180:
            return 'Normal'
        else:
            return 'Hyperglycemia'

    # Create a DataFrame for predictions starting 30 min after meal
    prediction_data = processed_data[time_mask_30min].copy()
    prediction_data['Zone'] = predictions.apply(glycemic_zone)

    # Create figure showing predictions vs ground truth
    fig = go.Figure()

    # Plot ground truth from meal time onwards
    fig.add_trace(go.Scatter(
        x=processed_data['datetime'],
        y=y_test,
        name='Ground Truth',
        line=dict(color='blue')
    ))

    # Plot predictions from 30 min onwards
    fig.add_trace(go.Scatter(
        x=prediction_data['datetime'],
        y=predictions,
        name='Predictions',
        line=dict(color='red')
    ))

    # Add shaded regions for glycemic zones
    fig.add_hrect(y0=0, y1=70, fillcolor="red", opacity=0.1, line_width=0)
    fig.add_hrect(y0=70, y1=180, fillcolor="green", opacity=0.1, line_width=0)
    fig.add_hrect(y0=180, y1=400, fillcolor="orange", opacity=0.1, line_width=0)

    # Add RMSE annotation
    fig.add_annotation(
        text=f"RMSE: {rmse:.2f} mg/dL",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=12),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )

    fig.update_layout(
        title='Glucose Level Predictions vs Ground Truth for the next two hours following the selected meal',
        xaxis_title='Time',
        yaxis_title='Glucose Level (mg/dL)',
        showlegend=True,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(size=10)
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("<h5>Feature Importances</h5>", unsafe_allow_html=True)

    # Add toggle for global vs local feature importances
    importance_type = st.radio("Select Feature Importance Type", ["Global", "Local"], key='importance_type')

    if importance_type == "Global":
        # Plot global feature importances from random forest
        fig, ax = plt.subplots()
        feature_importances.sort_values().plot(kind='barh', ax=ax)
        ax.set_xlabel('Feature Importance')
        ax.set_ylabel('Feature')
        plt.title("Global Feature Importances")
        st.pyplot(fig)
    else:
        # Allow user to select specific time point for local explanation
        time_options = processed_data[time_mask_30min]['datetime'].dt.strftime('%H:%M').tolist()
        selected_time = st.selectbox("Select time point for local explanation", time_options)

        # Get index of selected time
        selected_idx = processed_data.loc[time_mask_30min]['datetime'].dt.strftime('%H:%M').tolist().index(selected_time)

        # Use LIME to explain predictions at selected time point
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_test.values,
            feature_names=X_test.columns,
            class_names=['glucose'],
            discretize_continuous=True,
            mode='regression'
        )

        exp = explainer.explain_instance(
            X_test.loc[time_mask_30min].values[selected_idx],
            model.predict,
            num_features=10
        )

        # Extract feature contributions
        exp_list = exp.as_list()
        feature_names, contributions = zip(*exp_list)

        # Create DataFrame for plotting
        feature_importances_df = pd.DataFrame({
            'Feature': feature_names,
            'Contribution': contributions
        })
        feature_importances_df.sort_values('Contribution', inplace=True)

        # Plot bar chart
        fig, ax = plt.subplots()
        ax.barh(feature_importances_df['Feature'], feature_importances_df['Contribution'], color='skyblue')
        ax.set_xlabel('Contribution to Prediction')
        ax.set_title('Top 10 Features for Prediction at Selected Time Point')
        st.pyplot(fig)