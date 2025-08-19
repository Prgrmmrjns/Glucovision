import pandas as pd
import numpy as np
import os
from scipy.special import comb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import json
from params import *

def add_cumulative_features(glucose_data, combined_data):
    """Add cumulative features efficiently: time since last meal, last meal macros, cumulative insulin (2h)

    This implementation avoids per-row Python loops by using merge_asof joins and
    cumulative sums, reducing complexity from O(N*M) to roughly O(N log M).
    """
    # Ensure sorted inputs
    g = glucose_data.sort_values('datetime').reset_index()
    c = combined_data.sort_values('datetime').reset_index(drop=True)

    # Identify meal events (any row with non-zero macronutrients)
    macro_cols = [
        col for col in ['simple_sugars', 'complex_sugars', 'proteins', 'fats', 'dietary_fibers', 'carbohydrates']
        if col in c.columns
    ]

    # Last meal merge (backward asof)
    if macro_cols:
        meals = c.loc[c[macro_cols].sum(axis=1) > 0, ['datetime'] + macro_cols].copy()
        meals = meals.rename(columns={'datetime': 'last_meal_time'})
        asof_meal = pd.merge_asof(
            g[['index', 'datetime']].rename(columns={'datetime': 'dt'}).sort_values('dt'),
            meals.rename(columns={'last_meal_time': 'dt'}).sort_values('dt'),
            on='dt',
            direction='backward'
        ).rename(columns={'dt': 'datetime'})

        # Time since last meal (hours)
        time_since = (asof_meal['datetime'] - asof_meal['datetime'].where(asof_meal[macro_cols].notna().any(axis=1)))
        time_since_hours = time_since.dt.total_seconds().div(3600)
        g['time_since_last_meal'] = time_since_hours.fillna(24.0)

        # Last meal macros
        for col in macro_cols:
            g[f'last_meal_{col}'] = asof_meal[col].fillna(0.0)
    else:
        # If no macro columns exist, still provide defaults
        g['time_since_last_meal'] = 24.0

    # Cumulative insulin over last 2 hours via cum-sum differences with asof
    if 'insulin' in c.columns:
        ins = c[['datetime', 'insulin']].copy()
        ins['ins_cum'] = ins['insulin'].cumsum()

        # Value at t
        at_t = pd.merge_asof(
            g[['datetime']].rename(columns={'datetime': 'dt'}).sort_values('dt'),
            ins[['datetime', 'ins_cum']].rename(columns={'datetime': 'dt'}).sort_values('dt'),
            on='dt',
            direction='backward'
        )['ins_cum'].fillna(0.0)

        # Value at t-2h
        t_minus = g[['datetime']].copy()
        t_minus['dt'] = t_minus['datetime'] - pd.Timedelta(hours=2)
        at_tm2h = pd.merge_asof(
            t_minus[['dt']].sort_values('dt'),
            ins[['datetime', 'ins_cum']].rename(columns={'datetime': 'dt'}).sort_values('dt'),
            on='dt',
            direction='backward'
        )['ins_cum'].fillna(0.0)

        g['cumulative_insulin_2h'] = (at_t - at_tm2h).values

    # Restore original order and assign back into glucose_data
    out = glucose_data.copy()
    out.loc[g['index'], [col for col in g.columns if col not in ['index']]] = g.drop(columns=['index'])
    return out

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

def add_temporal_features(params, features, glucose_data, combined_data, prediction_horizon):
    """Add temporal features using Bezier curves (dataset-agnostic) with batched computation.

    Computes mapping in batches to avoid building a full (len(glucose) x len(combined))
    time-difference matrix, which can be very large. Also skips event rows where the
    source feature is zero to reduce work.
    """
    result = glucose_data.copy()
    g_times = result['datetime'].values.astype('datetime64[ns]').astype(np.int64)

    for feature in features:
        # Skip if source column not present
        if feature not in combined_data.columns:
            result[feature] = 0.0
            continue

        # Pre-filter combined data rows that matter for this feature
        src = combined_data[['datetime', feature]].copy()
        src = src[src[feature] != 0]
        if src.empty:
            result[feature] = 0.0
            continue

        src_times = src['datetime'].values.astype('datetime64[ns]').astype(np.int64)
        src_vals = src[feature].values.astype(float)

        curve = bezier_curve(np.array(params[feature]).reshape(-1, 2), num=32)
        x_curve, y_curve = curve[:, 0], curve[:, 1]
        max_h = float(x_curve[-1])

        mapped = np.zeros(len(result), dtype=float)
        # Process glucose timestamps in batches to limit memory
        batch_size = 2048
        for start in range(0, len(result), batch_size):
            end = min(start + batch_size, len(result))
            gt_batch = g_times[start:end]
            # Compute time differences (hours) for the batch
            td_hours = (gt_batch[:, None] - src_times[None, :]) / 3.6e12
            # Mask valid window [0, max_h]
            valid = (td_hours >= 0.0) & (td_hours <= max_h)
            if not valid.any():
                continue
            # Indices along curve for valid diffs
            idx = np.searchsorted(x_curve, td_hours[valid], side='left')
            idx = np.clip(idx, 0, len(y_curve) - 1)
            # Build weights matrix sparsely via zeros then fill valid positions
            weights = np.zeros_like(td_hours)
            weights[valid] = y_curve[idx]
            # Weighted sum over source events
            mapped[start:end] = weights.dot(src_vals)

        # Shift by prediction horizon to align target
        result[feature] = pd.Series(mapped, index=result.index).shift(-prediction_horizon)

    return result

def modify_time(glucose_data, target_hour):
    """Modify the time of day for all glucose data points while preserving date."""
    modified_data = glucose_data.copy()
    original_dates = modified_data['datetime'].dt.date
    original_minutes = modified_data['datetime'].dt.minute
    modified_data['datetime'] = pd.to_datetime([
        f"{date} {target_hour:02d}:{minute:02d}:00" 
        for date, minute in zip(original_dates, original_minutes)
    ])
    modified_data['hour'] = target_hour
    modified_data['time'] = target_hour + original_minutes / 60
    return modified_data

def train_and_predict(Xdf, idx_train, idx_val, target_col, feature_cols, test_batch, weights_train=None, weights_val=None, use_monotone=True):
    lgb_params = LGB_PARAMS.copy()
    if use_monotone:
        lgb_params['monotone_constraints'] = [MONOTONE_MAP.get(col, 0) for col in feature_cols]
    X_train = Xdf[feature_cols].iloc[idx_train].values
    y_train = Xdf[target_col].iloc[idx_train].values
    X_val = Xdf[feature_cols].iloc[idx_val].values
    y_val = Xdf[target_col].iloc[idx_val].values
    model = lgb.train(lgb_params, lgb.Dataset(X_train, label=y_train, weight=weights_train), valid_sets=[lgb.Dataset(X_val, label=y_val, weight=weights_val)])
    preds = model.predict(test_batch[feature_cols].values)
    return float(np.sqrt(mean_squared_error(test_batch[target_col].values, preds)))

def optimize_params(
    approach_name,
    features,
    fast_features,
    train_data,
    features_to_remove,
    prediction_horizon=12,
    n_trials=N_TRIALS,
):
    if LOAD_PARAMS:
        path = f"results/bezier_params/{approach_name}_bezier_params.json"
        if os.path.exists(path):
            return json.load(open(path))

    max_x_values = np.where(np.isin(features, fast_features), MAX_X_VALUES_FAST, MAX_X_VALUES_SLOW)

    def objective(trial):
        params = {}
        for i, f in enumerate(features):
            params[f] = [
                0.0, 0.0,
                trial.suggest_float(f"{f}_x2", 0.0, max_x_values[i]), trial.suggest_float(f"{f}_y2", 0.0, 1.0),
                trial.suggest_float(f"{f}_x3", 0.0, max_x_values[i]), trial.suggest_float(f"{f}_y3", 0.0, 1.0),
                trial.suggest_float(f"{f}_x4", 0.0, max_x_values[i]), 0.0,
            ]
        mapped_list = [add_temporal_features(params, features, g, c, prediction_horizon) for (g, c) in train_data]
        X_all = pd.concat([df[df.columns.difference(features_to_remove)] for df in mapped_list], ignore_index=True)
        y_all = pd.concat([df[f"glucose_{prediction_horizon}"] for df in mapped_list], ignore_index=True)
        full_df = X_all.copy()
        target_col = f"glucose_{prediction_horizon}"
        full_df[target_col] = y_all.values
        idx_train, idx_val = train_test_split(range(len(full_df)), test_size=VALIDATION_SIZE, random_state=RANDOM_SEED)
        rmse = train_and_predict(full_df, idx_train, idx_val, target_col, X_all.columns, full_df.iloc[idx_val])
        return rmse

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, n_jobs=N_JOBS, show_progress_bar=True)
    best = {f: [0.0, 0.0,
                study.best_params[f"{f}_x2"], study.best_params[f"{f}_y2"],
                study.best_params[f"{f}_x3"], study.best_params[f"{f}_y3"],
                study.best_params[f"{f}_x4"], 0.0] for f in features}
    json.dump(best, open(f"results/bezier_params/{approach_name}_bezier_params.json", 'w'), indent=2)
    return best

