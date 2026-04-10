# 🍎 Glucovision

Companion **research repository** for the paper on Elsevier **ScienceDirect**: [article PII S1532046425001741](https://www.sciencedirect.com/science/article/pii/S1532046425001741).

> Multimodal LLMs plus mechanistic Bézier temporal modeling for glucose forecasting from meal images in Type 1 Diabetes.

## 📊 Datasets

Glucovision uses two complementary datasets. **AZT1D is not in git** (see `.gitignore`); you must download it. **D1NAMO** raw CGM data comes from Zenodo; **meal macronutrients for the pipeline** are shipped as CSV snapshots in this repo for full reproducibility without any API.

### D1NAMO (primary)

1. **Raw records (CGM, insulin, meal images, `food.csv` timestamps)**  
   Download from [Zenodo record 5651217](https://zenodo.org/records/5651217) and extract at the **repository root** so you have:
   - `diabetes_subset_pictures-glucose-food-insulin/<patient>/glucose.csv`
   - `diabetes_subset_pictures-glucose-food-insulin/<patient>/insulin.csv`  
   Patient ids must match `PATIENTS_D1NAMO` in `scripts/params.py` (e.g. `001`, `002`, …).

2. **Macronutrients used by `scripts/` (Pixtral / mLLM estimates)**  
   The pipelines read **fixed CSVs** under `food_data/pixtral-large-latest/<patient>.csv` (columns include `datetime`, `simple_sugars`, `complex_sugars`, `proteins`, `fats`, `dietary_fibers`). **These files are intended to be version-controlled** so anyone can run `d1namo.py` and related scripts **without** calling Mistral.

3. **Optional: regenerate macronutrients from meal images**  
   If you want to reproduce or vary the vision step yourself:
   - Copy `.env.example` to `.env` at the repo root.
   - Set `MISTRAL_API_KEY` (create a key in the [Mistral AI console](https://console.mistral.ai)).
   - Run `food_annotations/food_annotations.ipynb` from the **repository root** (it uses `load_dotenv()`, model `pixtral-large-latest`, and writes `food_data/pixtral-large-latest/<patient>.csv`).  
   **Note:** API outputs are not guaranteed to be bit-identical to the committed CSVs (model updates, sampling, etc.). For **paper/exact reproduction**, use the committed `food_data` files.

### AZT1D (validation, local only)

1. Download the dataset from [Mendeley Data (gk9m674wcx)](https://data.mendeley.com/datasets/gk9m674wcx/1).
2. At the **repository root**, create this layout (names and spacing matter — this is what `scripts/azt1d.py` expects):

```
AZT1D 2025/
  CGM Records/
    Subject 1/Subject 1.csv
    Subject 2/Subject 2.csv
    …
```

3. Include every subject id listed in `PATIENTS_AZT1D` inside `scripts/params.py` (e.g. `1` … `25` with `14` omitted). If your archive uses different folder names, rename to match `Subject <n>/Subject <n>.csv`.

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


4. **Run analyses** (paths are resolved from the repo root regardless of your current directory)

### Core Evaluation
- `scripts/d1namo.py` - D1NAMO evaluation (Bezier vs baselines)
- `scripts/azt1d.py` - AZT1D evaluation
- `scripts/ablation_study.py` - Component contribution analysis
- `scripts/rmse_comparison.py` - RMSE tables and stats

### Feature Analysis
- `scripts/feature_importance.py` - Model interpretability
- `scripts/food_modifications.py` - Macronutrient sensitivity
- `scripts/time_impact.py` - Circadian effects
- `scripts/ga_vis.py` - Graphical abstract generation
- `scripts/combined_metabolic_vis.py` - Metabolic visualizations

## Citing

If you use this repository or the methods from the paper, please cite:

```bibtex
@article{wolber2025multimodal,
  title={Multimodal large language models and mechanistic modeling for glucose forecasting in type 1 diabetes patients},
  author={Wolber, JC and Samadi, ME and Sellin, Julia and Schuppert, Andreas},
  journal={Journal of Biomedical Informatics},
  pages={104945},
  year={2025},
  publisher={Elsevier}
}
```

## Acknowledgments

- **D1namo Dataset**: [Dubosson et al., 2018](https://doi.org/10.1016/j.imu.2018.09.003)
- **AZT1D Dataset**: [Khamesian et al., 2025](https://doi.org/10.17632/gk9m674wcx.1)

---
