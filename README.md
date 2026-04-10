# 🍎 Glucovision

> An innovative approach for leveraging meal images for glucose forecasting and patient metabolic modeling in Type 1 Diabetes

## 🚀 What is Glucovision?

Glucovision is a cutting-edge machine learning project that combines **multimodal Large Language Models (mLLMs)** with **mechanistic Bézier curve modeling** to predict blood glucose levels from meal images. By extracting macronutrient information directly from food photos, we enable automated glucose prediction without tedious manual food logging.

### 🎯 Key Features

- 🖼️ **Image-to-Prediction**: Transform meal photos into glucose forecasts
- 🤖 **mLLM Integration**: Automated macronutrient extraction using Pixtral Large
- 📈 **Temporal Modeling**: Optimized Bézier curves for nutrient absorption dynamics
- 🧠 **Cross-Patient Learning**: Learn from multiple patients with intelligent weighting
- ⏰ **Multiple Horizons**: Predict glucose changes at 30, 60, 90, and 120 minutes
- 📊 **Feature Importance**: Analysis of prediction drivers

## 📊 Datasets

Glucovision uses two complementary datasets. **AZT1D is not in git** (see `.gitignore`); you must download it. **D1NAMO** raw CGM data comes from Zenodo; **meal macronutrients for the pipeline** are shipped as CSV snapshots in this repo for full reproducibility without any API.

### 🔹 D1NAMO (primary)

1. **Raw records (CGM, insulin, meal images, `food.csv` timestamps)**  
   Download from [Zenodo record 5651217](https://zenodo.org/records/5651217) and extract at the **repository root** so you have:
   - `diabetes_subset_pictures-glucose-food-insulin/<patient>/glucose.csv`
   - `diabetes_subset_pictures-glucose-food-insulin/<patient>/insulin.csv`  
   Patient ids must match `PATIENTS_D1NAMO` in `analysis_scripts/params.py` (e.g. `001`, `002`, …).

2. **Macronutrients used by `analysis_scripts/` (Pixtral / mLLM estimates)**  
   The pipelines read **fixed CSVs** under `food_data/pixtral-large-latest/<patient>.csv` (columns include `datetime`, `simple_sugars`, `complex_sugars`, `proteins`, `fats`, `dietary_fibers`). **These files are intended to be version-controlled** so anyone can run `d1namo.py` and related scripts **without** calling Mistral.

3. **Optional: regenerate macronutrients from meal images**  
   If you want to reproduce or vary the vision step yourself:
   - Copy `.env.example` to `.env` at the repo root.
   - Set `MISTRAL_API_KEY` (create a key in the [Mistral AI console](https://console.mistral.ai)).
   - Run `food_annotations/food_annotations.ipynb` from the **repository root** (it uses `load_dotenv()`, model `pixtral-large-latest`, and writes `food_data/pixtral-large-latest/<patient>.csv`).  
   **Note:** API outputs are not guaranteed to be bit-identical to the committed CSVs (model updates, sampling, etc.). For **paper/exact reproduction**, use the committed `food_data` files.

### 🔹 AZT1D (validation, local only)

1. Download the dataset from [Mendeley Data (gk9m674wcx)](https://data.mendeley.com/datasets/gk9m674wcx/1).
2. At the **repository root**, create this layout (names and spacing matter — this is what `analysis_scripts/azt1d.py` expects):

```
AZT1D 2025/
  CGM Records/
    Subject 1/Subject 1.csv
    Subject 2/Subject 2.csv
    …
```

3. Include every subject id listed in `PATIENTS_AZT1D` inside `analysis_scripts/params.py` (e.g. `1` … `25` with `14` omitted). If your archive uses different folder names, rename to match `Subject <n>/Subject <n>.csv`.

4. Override path if needed: set environment variable `GLUCOVISION_AZT1D_DATA` to the absolute path of the `CGM Records` directory (the folder that directly contains `Subject 1`, `Subject 2`, …).

## 🛠️ Setup & Installation

1. **Clone the repository**
```bash
git clone https://github.com/Prgrmmrjns/Glucovision.git
cd Glucovision
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Data layout** (see the **Datasets** section above):
   - **D1NAMO Zenodo** → `diabetes_subset_pictures-glucose-food-insulin/` at repo root.
   - **D1NAMO meal macros for pipelines** → use tracked `food_data/pixtral-large-latest/*.csv`, or regenerate with `.env` + `MISTRAL_API_KEY` and `food_annotations/food_annotations.ipynb`.
   - **AZT1D** → `AZT1D 2025/CGM Records/Subject n/Subject n.csv` at repo root (not in git).

   Optional path overrides (absolute paths): `GLUCOVISION_D1NAMO_DATA`, `GLUCOVISION_FOOD_DATA`, `GLUCOVISION_AZT1D_DATA`, `GLUCOVISION_RESULTS`.

4. **Run analyses** (paths are resolved from the repo root regardless of your current directory)
```bash
python analysis_scripts/d1namo.py
python analysis_scripts/azt1d.py   # requires AZT1D data
# Optional: full paper-style outputs
python analysis_scripts/rmse_comparison.py
python analysis_scripts/ablation_study.py
```

## 🧪 Analysis Scripts

### Core Evaluation
- `analysis_scripts/d1namo.py` - D1NAMO evaluation (Bezier vs baselines)
- `analysis_scripts/azt1d.py` - AZT1D evaluation
- `analysis_scripts/ablation_study.py` - Component contribution analysis
- `analysis_scripts/rmse_comparison.py` - RMSE tables and stats

### Feature Analysis
- `analysis_scripts/feature_importance.py` - Model interpretability
- `analysis_scripts/food_modifications.py` - Macronutrient sensitivity
- `analysis_scripts/time_impact.py` - Circadian effects
- `analysis_scripts/ga_vis.py` - Graphical abstract generation
- `analysis_scripts/combined_metabolic_vis.py` - Metabolic visualizations

## 🔬 Technical Approach

### 1. mLLM Macronutrient Extraction
- **Pixtral Large** processes meal images
- Estimates: simple sugars, complex sugars, proteins, fats, dietary fibers
- Handles real-world food photography challenges

### 2. Mechanistic Modeling
- **Bézier curves** model temporal nutrient absorption
- **Global optimization** across all patients
- Physiologically-motivated temporal dynamics

### 3. Machine Learning Pipeline
- **LightGBM** gradient boosting regression
- **Patient weighting** (10:1 for target patient)
- **Temporal validation** with rolling windows

### 4. Multi-Dataset Validation
- **D1namo**: Primary mLLM validation (6 patients)
- **AZT1D**: Generalizability testing (25 patients)
- **Cross-dataset** insights on model robustness

## 📈 Results Highlights

- ✅ **Competitive RMSE**: 14.85 mg/dL (30min), 30.50 mg/dL (60min)
- 🎯 **Feature Evolution**: Glucose dominance → Time/macronutrient prominence
- 👥 **Patient Signatures**: Distinct metabolic profiles discovered
- 🕒 **Circadian Effects**: 13.4 mg/dL daily variation
- 🔄 **Reproducibility**: mLLM variability quantified (CV: 0.0-23.3%)

## 🏗️ Code Architecture

**Centralized Design** for maximum maintainability:
- `params.py` - All shared constants and parameters
- `processing_functions.py` - Core data processing functions
- **DRY Principle**: Zero code duplication across 11+ scripts

## 🤝 Contributing

We welcome contributions! Areas of interest:
- 🖼️ Alternative mLLM architectures
- 📊 New temporal modeling approaches
- 🎯 Additional validation datasets
- 🌐 Multi-language food recognition

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **D1namo Dataset**: [Dubosson et al., 2018](https://doi.org/10.1016/j.imu.2018.09.003)
- **AZT1D Dataset**: [Khamesian et al., 2025](https://doi.org/10.17632/gk9m674wcx.1)
- **Pixtral Large**: Mistral AI for multimodal capabilities

---

**Glucovision** — meal-image macronutrients and mechanistic modeling for glucose forecasting.