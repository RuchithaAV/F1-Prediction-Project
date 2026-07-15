#  Formula 1 Prediction Hub & Strategy Analytics

Welcome to the **Formula 1 Prediction Hub**, an end-to-end data science and machine learning project that simulates F1 race podiums, predicts pit stop strategies, and analyzes historical race telemetry. 

This repository leverages historical data from the **Ergast F1 Dataset** to train predictive models and features an interactive, premium-designed **Streamlit Web Application** for real-time race simulations and strategic insights.

---

##  Key Features

### 1.  Podium Simulator ([app.py](file:///d:/Data%20Science/projects/F1%20Prediction%20Project/app.py) / [train.py](file:///d:/Data%20Science/projects/F1%20Prediction%20Project/train.py))
* **Predictive Engine:** Uses a **Random Forest Classifier** trained on temporal historical F1 data (2000–2021) and validated on 2022+ seasons.
* **Feature Engineering:** Includes grid position, qualifying performance, gap to pole position, driver age, season cumulative points/wins (for drivers and constructors), and rolling recent form (podiums in the last 3 races).
* **Interactive Live Simulation:** Set up custom lineups or load actual historical race grids (2020–2024) to estimate the probability of a podium finish (P1, P2, or P3) for each driver on the grid.
* **Premium UI/UX:** High-fidelity custom-styled podium graphics, driver rankings, and detailed field probability reports.

### 2.  Pit Stop Lap Predictor
* **Tactical Forecasting:** Predicts the optimal lap for a driver's 1st or 2nd pit stop based on the circuit, starting grid, season, and total race distance.
* **Strategic Insights:** Automatically detects and displays tactical flags like **Aggressive Undercuts**, **Overcuts/Long Stints**, and **Standard Target Stints**.

### 3.  Advanced Analytics Notebooks
This project contains several Jupyter Notebooks for exploratory data analysis, unsupervised learning, and recommendation algorithms:
* **[F1_Podium_Prediction.ipynb](file:///d:/Data%20Science/projects/F1%20Prediction%20Project/notebooks/F1_Podium_Prediction.ipynb):** Prototype development and evaluation of the podium prediction classifier.
* **[Pit_stop_analysis.ipynb](file:///d:/Data%20Science/projects/F1%20Prediction%20Project/notebooks/Pit_stop_analysis.ipynb):** Exploration of historical pit stop durations, lap counts, and model prototyping for pit prediction.
* **[circuit_clustering.ipynb](file:///d:/Data%20Science/projects/F1%20Prediction%20Project/notebooks/circuit_clustering.ipynb):** K-Means clustering of F1 circuits based on characteristics like altitude, lap counts, average speed, and layout types.
* **[driver_clustering.ipynb](file:///d:/Data%20Science/projects/F1%20Prediction%20Project/notebooks/driver_clustering.ipynb):** PCA and clustering to identify driver profiles, career trajectories, and tier levels.

---

##  Repository Structure

```
├── f1 dataset/                   # Folder containing the Ergast CSV files
│   ├── circuits.csv
│   ├── constructors.csv
│   ├── drivers.csv
│   ├── races.csv
│   ├── results.csv
│   ├── qualifying.csv
│   ├── pit_stops.csv
│   └── ... (additional F1 metadata files)
├── models/                       # Pre-trained models and SafeLabelEncoder assets
│   ├── f1_podium_model.joblib
│   ├── f1_pit_stop_model.joblib
│   ├── le_driver.joblib
│   ├── le_constructor.joblib
│   └── ... (other classification encoders/models)
├── notebooks/                    # Jupyter notebooks for exploratory analysis
│   ├── F1_Podium_Prediction.ipynb
│   ├── Pit_stop_analysis.ipynb
│   ├── circuit_clustering.ipynb
│   └── driver_clustering.ipynb
├── app.py                        # Streamlit web application frontend & prediction logic
├── train.py                      # Training script for the Random Forest podium classifier
├── train_pit_stop.py             # CLI training script for the pit stop predictor
└── utils.py                      # Shared preprocessors, encoders, and helpers
```

---

##  Setup & Installation

### Prerequisites
Make sure you have Python 3.9+ installed on your system.

### 1. Clone the Repository
```bash
git clone <repository-url>
cd F1-Prediction-Project
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
Ensure you have the required packages:
```bash
pip install pandas numpy scikit-learn streamlit joblib matplotlib seaborn
```

---

##  Running the Project

### Start the Streamlit Web Application
To launch the interactive dashboard, run:
```bash
streamlit run app.py
```
This will open the prediction hub in your local browser (typically at `http://localhost:8501`).

### Retrain the Podium Model
If you make changes to the feature engineering process or update the datasets, you can retrain the model by running:
```bash
python train.py
```
This will evaluate the model on modern era races and overwrite `models/f1_podium_model.joblib`, `models/le_driver.joblib`, and `models/le_constructor.joblib` with fresh versions.

---

## 📈 Model Performance (`models/f1_podium_model.joblib`)
The Random Forest podium classifier is evaluated on 2022+ race data:
* **Accuracy:** ~80%+ (depending on split settings)
* **ROC-AUC Score:** ~0.85+
* **Key Features:** Grid position and qualifying gaps are the highest predictors, balanced by constructor strength and recent podium history.
