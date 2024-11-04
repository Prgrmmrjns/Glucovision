import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import pickle
import joblib
from app.text_and_functions import *

model = joblib.load("models/gpt4o/lightgbm_model_001.joblib")

with open('results/gpt4o.pkl', 'rb') as f:
    results = pickle.load(f)

params = results['001']['params']
feature_names = ['simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers', 'weight', 'fast_insulin', 'slow_insulin']

entered_own_data = False
image_analyzed = False
show_predictions = False
uploaded_file = None
window_size = 5

# Streamlit app
st.set_page_config(layout="wide", page_title="Glucose Forecasting")
st.title("Glucovision 🍽️ ")

# Styled markdown for description
st.markdown(
   glucovision_description, 
    unsafe_allow_html=True
)

# Add three examples from different patients from study with rmse
with st.sidebar:
    example_glucose_insulin = st.radio(
        "Blood glucose levels and insulin intake",('Use examples', 'Input my own data'))
    if example_glucose_insulin == 'Input my own data':
        glucose_levels = st.text_input("Enter the last 16 glucose values (interval 5 minutes):")
        entered_own_data = st.button('Enter data')
        slow_insulin = st.number_input("Slow Insulin (doses)", min_value=0, max_value=10, step=1)
        fast_insulin = st.number_input("Fast Insulin (doses)", min_value=0, max_value=10, step=1)
        
    else:
        glucose_levels = glucose_example_data[example_glucose_insulin]['glucose']
        slow_insulin = glucose_example_data[example_glucose_insulin]['slow_insulin']
        fast_insulin = glucose_example_data[example_glucose_insulin]['fast_insulin']

    selected_image = st.radio(
        "Choose image",
        ('Upload my own image', 'Salad', 'Various snacks', 'Glucose tablets')
    )
    hardcoded = st.checkbox("Use hardcoded values for example images")
    if not hardcoded:
        api_key = st.text_input("Enter your OPENAI API key here:")
    else: 
        image_analyzed = True
    # Style buttons
    if selected_image != 'Upload my own image':
        st.sidebar.markdown(
            f"<style> div[data-selected='true'] div[role='radiogroup'] label:nth-child({['Salad', 'Various snacks', 'Glucose tablets'].index(selected_image)+1}) {{ background-color: #f0f0f0; }} </style>",
            unsafe_allow_html=True)
    else:
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")

col1, col2, col3 = st.columns(3)
with col1:
    if (uploaded_file is not None or selected_image != 'Upload my own image'):
        if selected_image == 'Upload my own image':
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded Meal Image', width=250)
            temp_file_path = "temp_uploaded_image.jpg"
        else:
            img = Image.open(example_image_paths[selected_image])
            st.image(img, caption=f'{selected_image}', width=250)
            temp_file_path = example_image_paths[selected_image]
            
        img.save(temp_file_path)
        if not hardcoded and len(api_key) > 10:
            if st.button('Analyze image with multimodal LLM'):
                with st.spinner('Analyzing the image...'):
                    response = openai_api_call(api_key, temp_file_path, macronutrients_instruction)
                    message = response.json()['choices'][0]['message']['content']
                try:
                    parsed_info = parse_nutritional_info(message)
                    simple_sugars, complex_sugars, proteins, fats, dietary_fibers, weight = parsed_info[0], parsed_info[1], parsed_info[2], parsed_info[3], parsed_info[4], parsed_info[5] 
                    image_analyzed = True
                except:
                    st.error(response.json())
        if image_analyzed:
                simple_sugars, complex_sugars, proteins, fats, dietary_fibers, weight = macronutrient_example_data[selected_image]
                st.write("Macronutrients:")
                st.write(f"Simple sugars: {simple_sugars} grams")
                st.write(f"Complex sugars: {complex_sugars} grams")
                st.write(f"Dietary Fibers: {dietary_fibers} grams")
                st.write(f"Proteins: {proteins} grams")
                st.write(f"Fats: {fats} grams")
                st.write(f"Weight: {weight} grams")

if image_analyzed and len(glucose_levels) == 16 and st.button("Predict Future Glucose Levels"):
    show_predictions = True
    
    y_preds, X_test = predict_glucose_values(feature_names, params, model, window_size, glucose_levels, simple_sugars, complex_sugars, dietary_fibers, proteins, fats, weight, slow_insulin, fast_insulin)


    with col3:
        st.pyplot(plot_feature_importances(X_test, model))
    with col2:
        st.pyplot(plot_glucose_preds(glucose_levels, y_preds))
        

with col2:
    if not show_predictions and len(glucose_levels) == 16:
        st.pyplot(plot_prior_glucose_data(example_glucose_insulin, entered_own_data, glucose_levels))