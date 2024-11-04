import re
import numpy as np
import base64
import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta

food_data = pd.read_csv('gpt4o_food_data/food_data_001.csv')
patients_glucose_data = pd.read_csv('data/gpt4o/patients_glucose_data_001.csv')

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

glucose_example_data = {
    'Use examples': {
        'glucose': list(patients_glucose_data.iloc[0:16,0]),
        'slow_insulin': 3,
        'fast_insulin': 2
    }
}


macronutrient_example_data = {
    'Salad': food_data.loc[food_data['picture']=='008.jpg', ['simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers', 'weight']].iloc[0].tolist(),
    'Various snacks': food_data.loc[food_data['picture']=='026.jpg', ['simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers', 'weight']].iloc[0].tolist(),
    'Glucose tablets': food_data.loc[food_data['picture']=='012.jpg', ['simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers', 'weight']].iloc[0].tolist()
}

glucovision_description = """
<div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px;">
    <p style="font-size: 18px;">
        Glucovision is an application that helps people with Diabetes with Glucose management. 
        It analyzes meal images, insulin doses and past and current glucose values to forecast 
        glucose values up to 30 minutes in advance. Additionally, Glucovision displays which 
        macronutrients are most influential in affecting Glucose levels.
    </p>
</div>
"""

lifestyle_tips = """
    <div style="background-color: #f9f9f9; padding: 10px; border-radius: 5px; border-left: 5px solid #ffcc00;">
    <h3>💡 Tips for Managing Glucose Levels:</h3>
    <ul>
        <li>Monitor your glucose levels regularly.</li>
        <li>Maintain a balanced diet with appropriate macronutrients.</li>
        <li>Engage in regular physical activity.</li>
        <li>Consult with healthcare professionals for personalized advice.</li>
    </ul>
    </div>
"""

example_image_paths = {
    'Salad': 'example_images/008.jpg',
    'Various snacks': 'example_images/026.jpg',
    'Glucose tablets': 'example_images/012.jpg'
}

def plot_prior_glucose_data(example_glucose_insulin, entered_own_data, glucose_levels):
    time_steps = np.arange(-75, 5, 5)
    if example_glucose_insulin != 'Input my own data':
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(time_steps, glucose_levels, label='Glucose Levels', marker='o', color='blue')
        ax.set_xlabel('Time (5-minute intervals)')
        ax.set_ylabel('Glucose Level (mg/dL)')
        ax.legend()
    elif entered_own_data:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(time_steps, glucose_levels, label='Glucose Levels', marker='o', color='blue')
        ax.set_xlabel('Time (5-minute intervals)')
        ax.set_ylabel('Glucose Level (mg/dL)')
        ax.legend()
    return fig

def openai_api_call(api_key, temp_file_path, macronutrients_instruction):
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }
    base64_image = encode_image(temp_file_path)
    payload = {
    "model": "gpt-4o",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": macronutrients_instruction
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
    return response

def parse_nutritional_info(text):
    pattern = r'(Simple sugars \(g\)|Complex sugars \(g\)|Proteins \(g\)|Fats \(g\)|Dietary fibers \(g\)|Weight \(g\)):\s*(\d+)'
    matches = re.findall(pattern, text)
    nutritional_info = {match[0]: int(match[1]) for match in matches}
    simple_sugars = nutritional_info.get('Simple sugars (g)', 0)
    complex_sugars = nutritional_info.get('Complex sugars (g)', 0)
    proteins = nutritional_info.get('Proteins (g)', 0)
    fats = nutritional_info.get('Fats (g)', 0)
    dietary_fibers = nutritional_info.get('Dietary fibers (g)', 0)
    weight = nutritional_info.get('Weight (g)', 0)
    return simple_sugars, complex_sugars, proteins, fats, dietary_fibers, weight

def predict_glucose_values(feature_names, params, model, window_size, glucose_levels, simple_sugars, complex_sugars, dietary_fibers, proteins, fats, weight, slow_insulin, fast_insulin):

    now = datetime.now()
    time_values = [now - timedelta(minutes=5 * i) for i in range(16)]
    glucose_data = pd.DataFrame(glucose_levels, columns=["glucose"])
    glucose_data['datetime'] = [time.strftime('%Y-%m-%d %H:%M:%S') for time in time_values]
    glucose_data['glucose_change_5'] = glucose_data['glucose'] - glucose_data['glucose'].shift(-1)
    glucose_data['glucose_rolling_mean'] = glucose_data['glucose'].rolling(6).mean()
    glucose_data.dropna(inplace=True)

    combined_data = pd.DataFrame({
        'datetime': [now] ,
        'simple_sugars': simple_sugars,
        'complex_sugars': complex_sugars,
        'dietary_fibers': dietary_fibers,
        'proteins': proteins,
        'fats': fats,
        'weight': weight,
        'slow_insulin': slow_insulin,
        'fast_insulin': fast_insulin,
    })

    glucose_times = glucose_data['datetime'].values.astype('datetime64[s]').astype(np.int64)
    combined_times = combined_data['datetime'].values.astype('datetime64[s]').astype(np.int64)

    time_diff_hours = (glucose_times[:, None] - combined_times[None, :]) / 3600
    for feature_idx in range(len(feature_names)):
        decay, delay, multiplier, const, window = params[feature_idx*5:(feature_idx+1)*5]
        weights = (np.exp(-decay * np.abs(time_diff_hours - delay) ** 2) + const) * multiplier
        glucose_data[feature_names[feature_idx]] = np.dot(weights, combined_data.iloc[:, feature_idx + 1].values)

    X_test = glucose_data.drop('datetime', axis=1)
    y_preds = model.predict(X_test)
    
    cumsum_vec = np.cumsum(np.insert(y_preds, 0, 0))
    y_preds = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    return y_preds, X_test

def plot_glucose_preds(glucose_levels, y_preds):
    time_steps = np.arange(-75, 35, 5)
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(time_steps[:len(glucose_levels)], glucose_levels, label="Input Glucose Levels", marker='o', color='blue')
    ax1.plot(time_steps[len(glucose_levels)-1:], np.append(glucose_levels[-1], y_preds), label="Predicted Glucose Levels", linestyle='--', marker='o', color='blue')
    ax1.axvline(x=time_steps[len(glucose_levels)-1], color='gray', linestyle='--')
    ax1.set_xlabel("Time (minutes)")
    ax1.set_ylabel("Glucose Level (mg/dL)")
    ax1.legend()
    ax1.set_title("Glucose Level Forecasting with Macronutrients")
    ax1.grid(True)
    return fig

def plot_feature_importances(X_test, model):
    fig, ax2 = plt.subplots(figsize=(5, 3))
    importances_df = pd.DataFrame({
        'Features': X_test.columns,
        'Importances': model.feature_importances_
    })
    importances_df = importances_df.sort_values(by='Importances', ascending=False)
    ax2.barh(importances_df['Features'], importances_df['Importances'], color='skyblue')
    ax2.set_xlabel('Importance')
    ax2.invert_yaxis()
    ax2.set_title('Feature Importances')
    return fig

