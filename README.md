# FinSentry

### Predictive Cash-Out Risk Analysis for Cybercrime Transactions

FinSentry is a predictive analytics system that analyzes cybercrime-related
transaction networks to identify potential cash-out zones and time windows.

It combines **graph analysis, machine learning, risk scoring, and explainable
analysis** to detect suspicious transaction patterns.

## Key Features

- Transaction network analysis
- Known criminal network detection
- New pattern detection using Machine Learning
- Risk scoring
- Cash-out zone and time-window prediction
- Explainable risk analysis
- Interactive dashboard

## System Flow

```text
Transaction Data
       ↓
Data Processing
       ↓
Transaction Network Creation
       ↓
      ┌───────────────────────┐
      │                       │
      ↓                       ↓
Known Network Detection   New Pattern Detection
      │                       │
      └───────────┬───────────┘
                  ↓
          Machine Learning
             Risk Model
                  ↓
            Risk Scoring
                  ↓
        Cash-Out Risk Analysis
                  ↓
        Explainable Results
                  ↓
          Dashboard Display
```

## Technology Stack

**Frontend:** React.js, Vite, JavaScript  
**Backend:** Python, FastAPI  
**ML:** Random Forest, Joblib  
**Analysis:** Graph-based transaction analysis

## Project Structure

```text
FinSentry/
├── backend/
├── data/
├── frontend/
├── DEMO_SCRIPT.md
├── README.md
└── .gitignore
```

## How to Run

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend

Open a new terminal:

```bash
cd frontend
npm install
npm run preview
```

## Localhost

**Frontend:** http://localhost:4173/

**Backend:** http://127.0.0.1:8000

**API Documentation:** http://127.0.0.1:8000/docs

## Project Objective

To analyze suspicious transaction networks, detect known and emerging
patterns, calculate risk, and predict potential cash-out locations and
time windows through an explainable analytics dashboard.

## Disclaimer

This is an academic/prototype project developed for educational and
demonstration purposes using sample/synthetic data.