# PaperGrade-AI

PaperGrade AI is an AI-assisted paper quality prediction and decision support system designed for paper manufacturing. It combines a simulation engine, machine learning models, and an interactive dashboard to predict paper quality, explain predictions, and recommend operational actions.

## Features

* Paper quality prediction using trained machine learning models
* Interactive dashboard for visualization and analysis
* Simulation engine for paper grade transitions
* Decision engine for operational recommendations
* Prediction explanations for improved interpretability
* Historical data analysis and quality tracking

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

* Python
* Machine Learning
* Pandas
* Scikit-learn
* Streamlit (Dashboard)
* NumPy

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
