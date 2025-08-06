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
- 📊 **Feature Importance**: Real-time analysis of prediction drivers

## 🌟 Interactive Demo App

**Try our live Streamlit app!** 🎉

```bash
cd analysis_scripts
streamlit run app.py
```

The `app.py` provides an intuitive web interface where you can:
- 📱 Select patient data and meal images
- 🍎 View mLLM-estimated macronutrients
- 🎛️ Modify nutrient values interactively
- 📈 Get real-time glucose predictions
- 🧩 Explore feature importances and Bézier curves

## 📊 Datasets

Glucovision works with two complementary datasets:

### 🔹 D1namo Dataset (Primary)
- **6 Type 1 Diabetes patients** with meal images + CGM data
- **Essential for mLLM training and validation**
- Download from: [https://zenodo.org/records/5651217](https://zenodo.org/records/5651217) 
- ⚠️ **Important**: Click "Version 1.2.0" to get the complete dataset
- Extract to: `diabetes_subset_pictures-glucose-food-insulin/`

### 🔹 AZT1D Dataset (Validation)
- **25 Type 1 Diabetes patients** for model generalizability testing
- **Validates mechanistic modeling without images**
- Download from: [https://data.mendeley.com/datasets/gk9m674wcx/1](https://data.mendeley.com/datasets/gk9m674wcx/1)
- Extract to: `AZT1D 2025/`

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

3. **Download datasets** (see links above)

4. **Run the models**
```bash
# Train D1namo model
cd eval_scripts
python d1namo.py

# Train baseline comparison
python baseline.py

# Run ablation study
python ablation_study.py
```

5. **Launch interactive app** 🚀
```bash
cd analysis_scripts
streamlit run app.py
```

## 🧪 Analysis Scripts

### Core Evaluation
- `eval_scripts/d1namo.py` - Main mLLM-enhanced model
- `eval_scripts/baseline.py` - Baseline without mLLM features
- `eval_scripts/ablation_study.py` - Component contribution analysis
- `eval_scripts/approach_comparison.py` - Temporal mapping validation

### Feature Analysis
- `analysis_scripts/feature_importance.py` - Model interpretability
- `analysis_scripts/food_modifications.py` - Macronutrient sensitivity
- `analysis_scripts/combined_time_impact.py` - Circadian effects
- `analysis_scripts/combined_correlation_analysis.py` - Sugar-glucose relationships

### Visualization
- `analysis_scripts/app.py` - **🌟 Interactive Streamlit dashboard**
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

**🎯 Transform meal photos into glucose insights with Glucovision!** 📱➡️📈