# Cash-Out Risk Forecasting System

Predicts WHERE and WHEN cybercrime cash-out is likely, with plain-English reasoning —
built for a 48-hour hackathon demo.

## Project structure

```
data/          synthetic dataset + generator script
backend/       crime graph, ML risk model, explainability, FastAPI app
frontend/      React dashboard (dark console UI)
DEMO_SCRIPT.md exact walkthrough with verified real numbers for presenting
```

## How to run (two terminals)

**Terminal 1 — backend:**
```
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Check it's alive: open http://127.0.0.1:8000/docs

**Terminal 2 — frontend:**
```
cd frontend
npm install
npm run dev
```
Open http://localhost:5173

## Regenerating the dataset (optional)

If you want a fresh random dataset:
```
cd data
python3 generate_dataset.py
cd ../backend
python3 risk_model.py   # retrains the model on the new data
```

## How it works (pipeline order)

1. `data/generate_dataset.py` — builds a fake city (25 zones), complaints, mule account
   chains, and 6 hidden "networks" with repeat cash-out habits
2. `backend/crime_graph.py` — builds a graph of the whole dataset, matches new complaints
   against known networks by reused accounts
3. `backend/risk_model.py` — Random Forest model scores every zone for complaints that
   DON'T match a known network (85% top-3 hit rate on held-out data)
4. `backend/explainability.py` — turns either result into plain-English reasoning
5. `backend/main.py` — FastAPI app wiring it all together
6. `frontend/` — dashboard: zone heatmap, intake form, explanation panel, chain diagram,
   case log

See `DEMO_SCRIPT.md` for the exact presentation walkthrough with real, verified numbers.
