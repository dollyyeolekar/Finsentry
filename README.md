# FinSentry – Cash-Out Risk Forecasting System

FinSentry is a predictive analytics system designed to analyze cybercrime
transaction networks and identify potential cash-out zones and time windows.

It combines graph-based network analysis, machine learning, risk scoring,
and explainable analysis to support investigation of suspicious transaction patterns.

## Features

- Cybercrime transaction network analysis
- Known criminal network detection
- New-pattern detection using Machine Learning
- Zone-level risk scoring
- Cash-out zone prediction
- Explainable risk analysis
- Interactive dashboard

## Project Structure

```text
FinSentry/
├── backend/
│   ├── main.py
│   ├── crime_graph.py
│   ├── risk_model.py
│   ├── explainability.py
│   └── requirements.txt
│
├── data/
│   └── dataset files
│
├── frontend/
│   ├── dist/
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── DEMO_SCRIPT.md
├── README.md
└── .gitignore
Technologies
Backend
Python
FastAPI
Random Forest
Graph-based analysis
Joblib
Frontend
React
Vite
JavaScript
How to Run
Backend

Open a terminal:

cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

Backend API:

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/docs
Frontend

The repository currently contains the built frontend in frontend/dist.

To preview the built application:

cd frontend
npm install
npm run preview

Then open the localhost URL shown in the terminal.

### System Workflow

Transaction Data
       ↓
Graph Analysis
       ↓
Known Network Detection
       ↓
New Pattern Detection
       ↓
Machine Learning Risk Model
       ↓
Risk Scoring
       ↓
Cash-Out Zone Prediction
       ↓
Explainable Result
       ↓
Dashboard
Project Demo

See DEMO_SCRIPT.md for the complete demonstration workflow.

### Disclaimer

This project is an academic/hackathon prototype using synthetic data.
It is intended for research and demonstration purposes and is not a
production financial-crime detection system.



