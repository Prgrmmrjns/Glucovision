"""
Trainiert ein Bezier-basiertes Glukose-Prognose-Modell auf ALLEN D1NAMO-Patientendaten
und speichert die Modelle für die Streamlit-App (nächste 60 min: Horizonte 6, 9, 12).

Ausführung vom Projektroot: python analysis_scripts/train_d1namo_app_model.py
"""
import json
import os
import sys
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Projektroot = übergeordnetes Verzeichnis von analysis_scripts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from params import (
    DEFAULT_PREDICTION_HORIZON,
    FEATURES_TO_REMOVE_D1NAMO,
    FAST_FEATURES,
    FOOD_DATA_PATH,
    LGB_PARAMS,
    MONOTONE_MAP,
    N_TRIALS,
    OPTIMIZATION_FEATURES_D1NAMO,
    PATIENTS_D1NAMO,
    PREDICTION_HORIZONS,
    RANDOM_SEED,
    RESULTS_PATH,
    VALIDATION_SIZE,
)
from processing_functions import (
    add_temporal_features,
    get_d1namo_data,
    optimize_params,
)

# Horizonte für die App (30, 45, 60 min)
APP_HORIZONS = [6, 9, 12]
OUTPUT_DIR = os.path.join(RESULTS_PATH, "app_model")
RESULTS_DIR = RESULTS_PATH


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    features_to_remove = FEATURES_TO_REMOVE_D1NAMO + [
        f"glucose_{h}" for h in PREDICTION_HORIZONS
    ]

    # Nährstoffschätzungen: FOOD_DATA_PATH (Standard: repo/food_data/pixtral-large-latest)
    for p in PATIENTS_D1NAMO:
        food_csv = os.path.join(FOOD_DATA_PATH, f"{p}.csv")
        assert os.path.isfile(food_csv), f"Food-Datei fehlt: {food_csv}"
    print(f"Nährstoffdaten (Pixtral-Schätzungen): {FOOD_DATA_PATH}")

    # 1) Alle D1NAMO-Patientendaten laden (Glukose + Insulin aus D1NAMO, Mahlzeiten-Makros aus FOOD_DATA_PATH)
    print("Lade alle D1NAMO-Patientendaten …")
    patient_to_data = {}
    for patient in PATIENTS_D1NAMO:
        g_df, c_df = get_d1namo_data(patient)
        patient_to_data[patient] = (g_df, c_df)

    # 2) Bezier-Parameter auf allen Daten optimieren (wie generic in ablation_study)
    param_file = os.path.join(
        RESULTS_DIR, "bezier_params", "d1namo_generic_all_patients_bezier_params.json"
    )
    if os.path.exists(param_file):
        print("Lade vorhandene generische Bezier-Parameter …")
        with open(param_file, "r") as f:
            bezier_params = json.load(f)
    else:
        print("Optimiere Bezier-Parameter auf allen Patientendaten (kann dauern) …")
        os.makedirs(os.path.join(RESULTS_DIR, "bezier_params"), exist_ok=True)
        all_train_data = []
        for p in PATIENTS_D1NAMO:
            g_df, c_df = patient_to_data[p]
            train_days = g_df["datetime"].dt.day.unique()[:3]
            g_train = g_df[g_df["datetime"].dt.day.isin(train_days)]
            c_train = c_df[c_df["datetime"].dt.day.isin(train_days)]
            all_train_data.append((g_train, c_train))
        bezier_params = optimize_params(
            "d1namo_generic_all_patients",
            OPTIMIZATION_FEATURES_D1NAMO,
            FAST_FEATURES,
            all_train_data,
            features_to_remove,
            prediction_horizon=DEFAULT_PREDICTION_HORIZON,
            n_trials=N_TRIALS,
        )
        os.makedirs(os.path.dirname(param_file), exist_ok=True)
        with open(param_file, "w") as f:
            json.dump(bezier_params, f, indent=2)
        print("Generische Bezier-Parameter gespeichert.")

    # 3) Pro Horizon 6, 9, 12: Bezier-Features auf allen Daten, ein LightGBM trainieren, speichern
    feature_cols = None
    for horizon in APP_HORIZONS:
        target_col = f"glucose_{horizon}"
        print(f"Trainiere Modell für Horizon {horizon} (5-min-Intervalle) …")
        frames = []
        for p in PATIENTS_D1NAMO:
            g_df, c_df = patient_to_data[p]
            d = add_temporal_features(
                bezier_params,
                OPTIMIZATION_FEATURES_D1NAMO,
                g_df,
                c_df,
                prediction_horizon=horizon,
            )
            d["patient_id"] = f"patient_{p}"
            frames.append(d)
        full = pd.concat(frames, ignore_index=True)
        cols = [c for c in full.columns if c not in features_to_remove]
        if feature_cols is None:
            feature_cols = cols
        X = full[feature_cols]
        y = full[target_col]
        idx_train, idx_val = train_test_split(
            range(len(full)), test_size=VALIDATION_SIZE, random_state=RANDOM_SEED
        )
        lgb_params = LGB_PARAMS.copy()
        lgb_params["monotone_constraints"] = [
            MONOTONE_MAP.get(c, 0) for c in feature_cols
        ]
        model = lgb.train(
            lgb_params,
            lgb.Dataset(X.iloc[idx_train], label=y.iloc[idx_train]),
            valid_sets=[lgb.Dataset(X.iloc[idx_val], label=y.iloc[idx_val])],
        )
        rmse = float(np.sqrt(np.mean((y.iloc[idx_val] - model.predict(X.iloc[idx_val])) ** 2)))
        print(f"  Horizon {horizon}: Val-RMSE = {rmse:.4f}")
        joblib.dump(model, os.path.join(OUTPUT_DIR, f"model_{horizon}.joblib"))

    with open(os.path.join(OUTPUT_DIR, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "bezier_params.json"), "w") as f:
        json.dump(bezier_params, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "horizons.json"), "w") as f:
        json.dump(APP_HORIZONS, f)
    print(f"Modelle und Metadaten gespeichert in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
