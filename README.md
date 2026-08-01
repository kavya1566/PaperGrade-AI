# PaperGrade-AI

PaperGrade AI is an AI-assisted paper quality prediction and decision support system designed for paper manufacturing. It combines a simulation engine, machine learning models, and an interactive dashboard to predict paper quality, explain predictions, and recommend operational actions.

## Features

* Paper quality prediction using trained machine learning models
* Interactive dashboard for visualization and analysis
* Simulation engine for paper grade transitions
* Decision engine for operational recommendations
* Prediction explanations for improved interpretability
* Historical data analysis and quality tracking

## System Architecture

PaperGrade-AI follows a modular decision-support architecture that combines machine learning, explainability, simulation, and historical process intelligence to assist operators during paper grade transitions.

The workflow begins with operator inputs, processes them through specialized AI services, validates recommendations using simulation and historical evidence, and presents an integrated decision package through an interactive dashboard.

![System Workflow](image.png)

## Project Structure

```text
paper-grade-ai/
├── dashboard/          # Dashboard application
├── ml/                 # Machine learning pipeline
│   ├── artifacts/      # Model metadata and artifacts
│   └── datasets/       # Training datasets
├── services/           # Business logic and services
├── simulator/          # Paper grade transition simulator
├── requirements.txt
└── README.md
```

## Technology Stack

| Category         | Technologies               |
| ---------------- | -------------------------- |
| Language         | Python                     |
| Machine Learning | Scikit-learn, XGBoost      |
| Data Processing  | Pandas, NumPy              |
| Dashboard        | Streamlit                  |
| Visualization    | Matplotlib, Plotly         |
| Explainability   | SHAP                       |
| Simulation       | Custom Digital Twin Engine |


## Installation

Clone the repository:

```bash
git clone https://github.com/kavya1566/PaperGrade-AI.git
cd PaperGrade-AI
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

**Windows**

```bash
.venv\Scripts\activate
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Start the dashboard:

```bash
streamlit run dashboard/app.py
```

If your project uses a different entry point, update the command accordingly.

## Components

### Dashboard

Provides an interactive interface for:

* Quality prediction
* Visualization
* Recommendation display
* Model explanations

### Machine Learning

Responsible for:

* Data preprocessing
* Feature engineering
* Model inference
* Prediction pipeline

### Simulator

Simulates paper grade transitions and generates operational scenarios for testing and evaluation.

### Services

Contains the application's business logic, including:

* Prediction service
* Recommendation service
* Explanation service
* Decision engine
* History service

## Model Performance

The PaperGrade AI prediction engine is built using an **XGBoost classifier** trained to predict off-spec paper grade transitions from the **first six minutes** of process data. This enables early detection of quality issues before the transition is complete.

To better represent real industrial operating conditions, the training pipeline introduces:

* **10% label noise** to simulate operator labelling inconsistencies and borderline transition classifications.
* **Gaussian feature noise (σ = 0.12)** to represent sensor uncertainty, calibration drift, and process variability.
* Feature selection to retain the most informative variables.
* Probability calibration for reliable risk estimates.
* F1-score threshold optimisation to determine the operational decision threshold.

### Hold-Out Test Performance

| Metric    |     Score |
| --------- | --------: |
| Accuracy  | **88.1%** |
| Precision | **89.6%** |
| Recall    | **70.8%** |
| F1 Score  | **79.1%** |
| ROC AUC   | **84.8%** |

These results were obtained using a **held-out test set of 1,500 paper grade transitions (15% of the dataset)** that was not used during model training or threshold optimisation.

### Operational Characteristics

* Early prediction using the **first six minutes** of transition data.
* Predicts the probability of an off-spec paper grade transition.
* Uses an **F1-optimised decision threshold (0.47)** for operational decision-making.
* Provides explainable predictions together with recommended operator actions through the Decision Intelligence framework.

## Dashboard

### Prediction & Recommendation Dashboard
Predict paper quality for a selected grade transition and receive AI-assisted operational recommendations.
![Prediction & Recommendation Dashboard](<prediction-and-recommendations.png>)

### AI Decision Support & Operational Recommendations
Review AI-generated recommendations, process health indicators, causal explanations, and suggested operational adjustments before applying changes to production.
![AI Decision Support](ai-decision-support.png)

### Explainable AI & Future State Prediction
Interpret model predictions using SHAP-based feature attribution, understand the physical causal chain behind each recommendation, and forecast future process behavior if no corrective actions are applied.
![Explainable AI & Future State Prediction](explainable-ai-future-state.png)

### Historical Transition Analysis
Compare the current grade transition with historical production runs and analyze process telemetry correlations to identify key relationships influencing paper quality.
![Historical Transition Analysis](historical-transition-analysis.png)

### Digital Twin Sandbox
Interactively simulate grade transitions by modifying process parameters, evaluating predicted process behavior, and validating operating strategies before deployment to the production line.
![Digital Twin Sandbox](digital-twin-sandbox.png)

### Simulation Results & Process Forecast
Visualize the predicted process behavior over time, evaluate quality deviations and stabilization metrics, and assess whether the proposed operating parameters will achieve the target paper grade before deployment.
![Simulation Results](simulation-results-1.png)
![Simulation Results](simulation-results-2.png)
![Simulation Results](simulation-results-3.png)
![Simulation Results](simulation-results-4.png)

### Feedback & Audit Log
Track operator decisions, review historical transition records, monitor model performance, and maintain a complete audit trail to support continuous process improvement and operational transparency.
![Feedback & Audit Log](feedback-audit-log.png)

![Engineering Console & Model Governance](engineering-console-model-governance.png)


## Repository

```text
PaperGrade-AI
├── Dashboard
├── ML Pipeline
├── Simulation Engine
├── Decision Engine
└── Recommendation System
```

## Future Improvements

* Real-time production data integration
* Model retraining pipeline
* REST API support
* Docker deployment
* Performance monitoring
* User authentication

## License

This project is intended for educational and research purposes.
