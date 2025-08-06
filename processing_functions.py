"""
Shared processing functions for Glucovision project
"""

import pandas as pd
import numpy as np
import os
from scipy.special import comb
from params import *

def bezier_curve(points, num=50):
    """Generate Bezier curve from control points using Bernstein polynomials"""
    points = np.array(points).reshape(-1, 2)
    points[0] = [0.0, 0.0]
    control_points = points[1:].copy()
    sorted_indices = np.argsort(control_points[:, 0])
    points[1:] = control_points[sorted_indices]
    points[-1, 1] = 0.0
    n = len(points) - 1
    t = np.linspace(0, 1, num)
    curve = np.zeros((num, 2))
    for i, point in enumerate(points):
        curve += np.outer(comb(n, i) * (t**i) * ((1-t)**(n-i)), point)
    return curve[np.argsort(curve[:, 0])]

def get_projected_value(window, prediction_horizon):
    """Project future value using polynomial regression"""
    x = np.arange(len(window))
    coeffs = np.polyfit(x, window, deg=3)
    return np.polyval(coeffs, len(window) + prediction_horizon)

def get_d1namo_data(patient):
    """Load D1namo data for a patient"""
    glucose_data = pd.read_csv(f"{D1NAMO_DATA_PATH}/{patient}/glucose.csv")
    insulin_data = pd.read_csv(f"{D1NAMO_DATA_PATH}/{patient}/insulin.csv")
    food_data = pd.read_csv(f"{FOOD_DATA_PATH}/{patient}.csv")
    
    glucose_data["datetime"] = pd.to_datetime(glucose_data["date"] + ' ' + glucose_data["time"])
    glucose_data = glucose_data.drop(['type', 'comments', 'date', 'time'], axis=1)
    glucose_data['glucose'] *= GLUCOSE_CONVERSION_FACTOR
    glucose_data['hour'] = glucose_data['datetime'].dt.hour
    glucose_data['time'] = glucose_data['hour'] + glucose_data['datetime'].dt.minute / 60
    
    insulin_data["datetime"] = pd.to_datetime(insulin_data["date"] + ' ' + insulin_data["time"])
    insulin_data.fillna(0, inplace=True)
    insulin_data['insulin'] = insulin_data['slow_insulin'] + insulin_data['fast_insulin']
    insulin_data = insulin_data.drop(['slow_insulin', 'fast_insulin', 'comment', 'date', 'time'], axis=1)
    
    food_data['datetime'] = pd.to_datetime(food_data['datetime'], format='%Y:%m:%d %H:%M:%S')
    food_data = food_data[['datetime', 'simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers']]
    
    combined_data = pd.concat([food_data, insulin_data]).sort_values('datetime').reset_index(drop=True)
    combined_data.fillna(0, inplace=True)
    
    for horizon in PREDICTION_HORIZONS:
        glucose_data[f'glucose_{horizon}'] = glucose_data['glucose'].shift(-horizon) - glucose_data['glucose']
    
    glucose_data['glucose_change'] = glucose_data['glucose'] - glucose_data['glucose'].shift(1)
    glucose_data['glucose_change_projected'] = glucose_data['glucose_change'].rolling(6, min_periods=6).apply(lambda window: get_projected_value(window, 6))
    glucose_data['glucose_projected'] = glucose_data['glucose'].rolling(6, min_periods=6).apply(lambda window: get_projected_value(window, 6))
    glucose_data.dropna(subset=[f'glucose_24'], inplace=True)
    glucose_data['patient_id'] = patient
    
    return glucose_data, combined_data

def load_azt1d_data():
    """Load AZT1D data for all patients"""
    data_list = []
    
    for patient in PATIENTS_AZT1D:
        file_path = f"{AZT1D_DATA_PATH}/Subject {patient}/Subject {patient}.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['patient'] = patient
            df['datetime'] = pd.to_datetime(df[AZT1D_COLUMNS['datetime']])
            df['glucose'] = df[AZT1D_COLUMNS['glucose']].fillna(0)
            df['carbohydrates'] = df[AZT1D_COLUMNS['carbohydrates']].fillna(0)
            df['insulin'] = df[AZT1D_COLUMNS['insulin']].fillna(0)
            
            # Add hour and time features
            df['hour'] = df['datetime'].dt.hour
            df['time'] = df['hour'] + df['datetime'].dt.minute / 60
            
            # Keep only needed columns
            df = df[['patient', 'datetime', 'glucose', 'carbohydrates', 'insulin', 'hour', 'time']].copy()
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # Add prediction horizon features
            for horizon in PREDICTION_HORIZONS:
                df[f'glucose_{horizon}'] = df['glucose'].shift(-horizon) - df['glucose']
            
            # Add glucose change and projected features
            df['glucose_change'] = df['glucose'] - df['glucose'].shift(1)
            df['glucose_change_projected'] = df['glucose_change'].rolling(6, min_periods=6).apply(lambda window: get_projected_value(window, 6))
            df['glucose_projected'] = df['glucose'].rolling(6, min_periods=6).apply(lambda window: get_projected_value(window, 6))
            
            df = df.dropna(subset=[f'glucose_24'])
            
            if len(df) > 100:  # Only include patients with sufficient data
                data_list.append(df)
    
    return data_list

def add_d1namo_features(params, features, data):
    """Add D1namo temporal features using Bezier curves"""
    glucose_data, combined_data = data
    glucose_data = glucose_data.copy()
    time_diff_hours = (glucose_data['datetime'].values.astype(np.int64)[:, None] - combined_data['datetime'].values.astype(np.int64)[None, :]) / 3600000000000
    base_curves = {f: bezier_curve(np.array(params[f]).reshape(-1, 2), num=50) for f in features if f in params}
    
    for i, feature in enumerate(features):
        curve = base_curves[feature]
        x_curve, y_curve = curve[:, 0], curve[:, 1]
        valid_mask = (time_diff_hours >= 0) & (time_diff_hours <= x_curve[-1])
        weights = np.zeros_like(time_diff_hours)
        weights[valid_mask] = y_curve[np.clip(np.searchsorted(x_curve, time_diff_hours[valid_mask]), 0, len(y_curve) - 1)]
        glucose_data[feature] = np.dot(weights, combined_data[feature])
    
    return glucose_data

def add_azt1d_features(params, features, glucose_data, combined_data):
    """Add AZT1D temporal features using Bezier curves"""
    glucose_data = glucose_data.copy()
    
    time_diff_hours = (glucose_data['datetime'].values.astype(np.int64)[:, None] - 
                      combined_data['datetime'].values.astype(np.int64)[None, :]) / 3600000000000
    
    for feature in features:
        if feature in params:
            curve = bezier_curve(params[feature], num=50)
            x_curve, y_curve = curve[:, 0], curve[:, 1]
            
            valid_mask = (time_diff_hours >= 0) & (time_diff_hours <= x_curve[-1])
            weights = np.zeros_like(time_diff_hours)
            weights[valid_mask] = y_curve[np.clip(np.searchsorted(x_curve, time_diff_hours[valid_mask]), 0, len(y_curve) - 1)]
            
            feature_values = combined_data[feature].values
            feature_result = np.dot(weights, feature_values)
            glucose_data[feature] = feature_result
    
    return glucose_data

def modify_time(glucose_data, target_hour):
    """Modify the time of day for all glucose data points while preserving date."""
    modified_data = glucose_data.copy()
    # Set hour to target_hour while preserving date and minute
    original_dates = modified_data['datetime'].dt.date
    original_minutes = modified_data['datetime'].dt.minute
    modified_data['datetime'] = pd.to_datetime([
        f"{date} {target_hour:02d}:{minute:02d}:00" 
        for date, minute in zip(original_dates, original_minutes)
    ])
    modified_data['hour'] = target_hour
    modified_data['time'] = target_hour + original_minutes / 60
    return modified_data

# Backward compatibility aliases
get_data = get_d1namo_data  # For scripts that use get_data
add_features = add_d1namo_features  # For scripts that use add_features