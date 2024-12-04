import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import base64
import re
import requests
import itertools
from warnings import simplefilter, filterwarnings


# Suppress warnings for cleaner output
filterwarnings("ignore", category=FutureWarning)
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

macronutrients_instruction = '''Examine the provided meal image to analyze and estimate its nutritional content accurately. Focus on determining the amounts of simple sugars (like industrial sugar and honey), 
complex sugars (such as starch and whole grains), proteins, fats, and dietary fibers (found in fruits and vegetables), all in grams. Also estimate the total weight of the meal in grams.
To assist in accurately gauging the scale of the meal, a 1 Swiss Franc coin, which has a diameter of 23.22 mm, may be present in the picture. 
Use the size of this coin as a reference to estimate the size of the meal and the amounts of the nutrients more precisely. 
Provide your assessment of each nutritional component in grams. All estimates should be given as a single whole number. If there is no coin in the picture or the meal is covered partially, estimate anyways.
Format your response as follows:
- Simple sugars (g): 
- Complex sugars (g): 
- Proteins (g): 
- Fats (g): 
- Dietary fibers (g): 
- Weight (g): 
- Explanation: 

Example response:
Simple sugars (g): 40
Complex sugars (g): 60
Proteins (g): 25
Fats (g): 30
Dietary fibers (g): 5 
Weight (g): 750
Explanation: The pizza and cola meal, with its refined crust and toppings, is rich in carbs, fats, and proteins. The cola boosts the meal's simple sugars. 
The 1 Swiss Franc coin helps estimate the pizza at 30 cm diameter and the cola at 330 ml, indicating a significant blood sugar impact.'''

PATIENTS = ['001', '002', '004', '006', '007', '008']

# Features used in the model
features = ['simple_sugars', 'complex_sugars', 'fats', 'dietary_fibers', 'proteins', 'fast_insulin', 'slow_insulin']



# Metabolism parameters for features (sensitivity and peak time)
feature_params = {
    'simple_sugars': [0.4, 0.5],  # [metabolism_rate_param, peak_time]
    'complex_sugars': [0.3, 0.5],
    'proteins': [0.2, 3.5],
    'fats': [0.05, 3.5], 
    'dietary_fibers': [0.05, 3.5],
    'fast_insulin': [1.0, 0.5], 
    'slow_insulin': [0.5, 1.0]
}

model = 'gpt4o'

def get_projected_value(window, prediction_horizon):
    x = np.arange(len(window))
    coeffs = np.polyfit(x, window, deg=3)
    poly = np.poly1d(coeffs)
    projected_value = poly(len(window) + prediction_horizon)
    return projected_value

def get_available_images(patient):
    """
    Get list of available food images for a patient.
    """
    image_dir = f"diabetes_subset_pictures-glucose-food-insulin/{patient}/food_pictures"
    if os.path.exists(image_dir):
        return [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    return []

def add_features(params, features, prediction_horizon):
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

def prepare_glucose_data(glucose_data, prediction_horizon):
    # Prepare glucose data
    glucose_data['hour'] = glucose_data['datetime'].dt.hour

    glucose_data['glucose_next'] = glucose_data['glucose'] - glucose_data['glucose'].shift(-prediction_horizon)

    glucose_data['glucose_change'] = glucose_data['glucose'] - glucose_data['glucose'].shift(1)

    glucose_data[f'glucose_change_sh_3'] = glucose_data['glucose_change'].shift(3)

    for window in [2, 3, 6]:
        glucose_data[f'glucose_change_std_{window}'] = glucose_data['glucose_change'].rolling(window=window).std()
    
    delta = glucose_data['glucose'].diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=6).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=6).mean()
    glucose_data['glucose_rsi'] = 100 - (100 / (1 + gain / loss))

    glucose_data['glucose_change_projected'] = glucose_data['glucose_change'].rolling(
        window=6, min_periods=6
    ).apply(lambda window: get_projected_value(window, prediction_horizon))
    glucose_data['glucose_projected'] = glucose_data['glucose'].rolling(
        window=6, min_periods=6
    ).apply(lambda window: get_projected_value(window, prediction_horizon))
    glucose_data.dropna(subset=['glucose_next'], inplace=True)
    return glucose_data, combined_data

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def openai_api_call(base64_image, llm_instructions, api_key):
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }
    payload = {
    "model": "gpt-4o",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": llm_instructions
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    try:
        message = response.json()['choices'][0]['message']['content']
        return message
    except (KeyError, IndexError):
        return response.json()['error']['message']

def meal_modification_suggestions(processed_data, model, X_test, llm_estimations):
    st.markdown("<h5>Meal Modification Suggestions</h5>", unsafe_allow_html=True)

    # Get the time point with the highest predicted glucose spike
    time_mask = processed_data['datetime'] >= meal_time
    max_spike_index = (processed_data.loc[time_mask]['glucose'] - processed_data.loc[time_mask]['glucose_next']).idxmax()
    hyperglycemia_point = processed_data.loc[max_spike_index]
    
    # Only show suggestions if predicted glucose is in hyperglycemic range (>180 mg/dL)
    predicted_glucose = hyperglycemia_point['glucose'] + hyperglycemia_point['glucose_next']
    if predicted_glucose <= 180:
        st.write("No meal modifications needed - predicted glucose levels are within normal range.")
        return

    # Extract the meal features
    meal_features = ['simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers']
    original_meal = X_test.loc[max_spike_index, meal_features]

    # Define increments for grid search
    adjustments = [-20, -10, 0, 10, 20]

    # Initialize a DataFrame to store results
    results = []

    # Perform grid search over combinations of meal feature modifications
    for deltas in itertools.product(adjustments, repeat=len(meal_features)):
        modified_meal = original_meal.copy()
        for i, feature in enumerate(meal_features):
            modified_meal[feature] += deltas[i]
        modified_meal = modified_meal.clip(lower=0)

        # Prepare the modified input
        X_modified = X_test.loc[max_spike_index].copy()
        X_modified.update(modified_meal)
        X_modified = X_modified.values.reshape(1, -1)

        # Predict the glucose level
        predicted_glucose = X_modified[0][X_test.columns.get_loc('glucose')] - model.predict(X_modified)[0]

        # Store the results
        result = modified_meal.to_dict()
        result['predicted_glucose'] = predicted_glucose
        result['modifications'] = {meal_features[i]: deltas[i] for i in range(len(meal_features))}
        results.append(result)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Find the best modification (lowest predicted glucose)
    best_modification = results_df.loc[results_df['predicted_glucose'].idxmin()]

    # Display the results
    st.write("### Suggested Meal Modifications:")
    st.write(f"To reduce your predicted glucose spike from **{hyperglycemia_point['glucose'] - hyperglycemia_point['glucose_next']:.2f} mg/dL** to **{best_modification['predicted_glucose']:.2f} mg/dL**, consider making the following changes to your meal:")

    # Extract modifications
    modifications = {k.replace('_', ' ').title(): int(v) for k, v in best_modification['modifications'].items() if v != 0}
    
    # Prepare the LLM prompt
    if 'api_key' in globals():
        base64_image = encode_image(image_path)
        modifications_text = '\n'.join([f"- {k}: {v:+d}g" for k, v in modifications.items()])
        llm_instructions = f"""Based on the meal shown in the image, provide at maximum 3 specific, actionable suggestions to adjust this meal to reduce glucose spike.

**Original Estimated Nutritional Values:**
{llm_estimations}

**Recommended Changes:**
{modifications_text}

Please provide concrete suggestions based on what you see in the image. For example:
- If you see rice, suggest reducing the portion or replacing with cauliflower rice
- If you see bread, suggest whole grain alternatives
- If you see pasta, suggest adding more vegetables or protein

Format each suggestion as:
1. What specific item in the meal to change
2. How exactly to change it (reduce portion, substitute, or add something)
3. The expected benefit for glucose control"""
        
        suggestion = openai_api_call(base64_image, llm_instructions, api_key)
        st.write(suggestion)


st.markdown("<h3 style='margin-top:-2rem;'>Glucovision: Glucose Level Forecasting Using Multimodal LLMs 👀</h3>", unsafe_allow_html=True)
st.markdown("""
<small>Welcome to Glucovision, a framework designed to forecast glucose levels in Type 1 Diabetes patients by incorporating meal image information using multimodal LLMs.

**Note:** This app is a demonstration using the D1namo dataset and is not intended for medical advice.</small>
""", unsafe_allow_html=True)

submit_button = st.button("Generate Predictions")

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

    # Show preview image before submit
    image_path = f"diabetes_subset_pictures-glucose-food-insulin/{selected_patient}/food_pictures/{selected_image}"
    st.image(image_path, caption="Preview", width=100)

    api_key = st.text_input("Enter your OPENAI API key (optional)", type="password")

    if not api_key:
        st.warning("Enter your OpenAI API key first!")
        st.stop()

    prediction_horizon = st.selectbox(
        "Prediction Horizon",
        [6, 12],
    )

if submit_button and api_key:
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

    food_data, insulin_data, glucose_data = load_data(selected_patient, model)
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

    selected_food_data = food_data[food_data.index <= food_data[food_data['picture'] == selected_image].index[0]]
    meal_time = selected_food_data['datetime'].iloc[-1]
    message = selected_food_data['message'].iloc[-1]

    selected_day =  int(meal_time.strftime('%d'))
    
    # Show preview in sidebar
    image_path = f"diabetes_subset_pictures-glucose-food-insulin/{selected_patient}/food_pictures/{selected_image}"
    with st.sidebar:
        st.image(image_path, caption="Preview", width=100)
        
    st.markdown("<h5>Selected meal image</h5>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image_path, caption=f"Patient {selected_patient} - {selected_image}", width=200)

    with col2:
        with st.expander("View LLM Reasoning"):
            st.write(message)
        with st.expander("View LLM Estimations"):
            llm_estimations = ""
            for feature in features:
                if feature == 'fast_insulin' or feature == 'slow_insulin':
                    continue
                llm_estimations += f"{feature}: {selected_food_data[feature].iloc[-1]}\n"
            st.write(llm_estimations)

    # Call the function
    glucose_data, combined_data = prepare_glucose_data(glucose_data, prediction_horizon)

    # Add features
    processed_data = add_features(feature_params, features, prediction_horizon)
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
    model = joblib.load(f'models/{model}/6_{selected_day}_{selected_patient}.joblib')

    feature_importances = pd.Series(model.feature_importances_, index=X_test.columns)

    # Add tabs for the outputs
    st.markdown("<h5>Results</h5>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Predictions", "Feature Importances", "Meal Modification Suggestions"])

    with tab1:
        st.markdown("<h5>Predictions</h5>", unsafe_allow_html=True)

        # Filter predictions starting 30 minutes after meal time
        time_mask = processed_data['datetime'] >= meal_time + pd.Timedelta(minutes=prediction_horizon*5)
        predictions = model.predict(X_test[time_mask])
        predictions = processed_data[time_mask]['glucose'] - predictions

        # Check for hyper/hypoglycemic predictions
        hyper_mask = predictions > 180
        hypo_mask = predictions < 70

        if any(hyper_mask):
            hyper_times = processed_data.loc[time_mask][hyper_mask]['datetime'].dt.strftime('%H:%M').tolist()
            st.warning(f"⬆️ 🔴 High glucose predicted at: {', '.join(hyper_times)}")
        elif any(hypo_mask):
            hypo_times = processed_data.loc[time_mask][hypo_mask]['datetime'].dt.strftime('%H:%M').tolist() 
            st.warning(f"⬇️ 💙 Low glucose predicted at: {', '.join(hypo_times)}")
        else:
            st.success("✨ 🎯 Glucose levels are predicted to stay in safe range for all timepoints!")

        # Calculate RMSE for predictions 30 minutes onwards
        rmse = np.sqrt(np.mean((predictions - y_test[time_mask]) ** 2))

        # Define glycemic zones
        def glycemic_zone(glucose_level):
            if glucose_level < 70:
                return 'Hypoglycemia'
            elif 70 <= glucose_level <= 180:
                return 'Normal'
            else:
                return 'Hyperglycemia'

        # Create a DataFrame for predictions starting 30 min after meal
        prediction_data = processed_data[time_mask].copy()
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

        # Plot global feature importances from random forest
        fig, ax = plt.subplots()
        feature_importances.sort_values().plot(kind='barh', ax=ax)
        ax.set_xlabel('Feature Importance')
        ax.set_ylabel('Feature')
        plt.title("Global Feature Importances")
        st.pyplot(fig)
    with tab3:
    # Call the new function for meal modification suggestions
        meal_modification_suggestions(processed_data, model, X_test, llm_estimations)
