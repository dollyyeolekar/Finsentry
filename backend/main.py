"""
Backend API
=============
Wires together:
  - crime_graph.py     (network matching)
  - risk_model.py       (ML zone risk scoring for novel cases)
  - explainability.py   (plain-English reasoning)

into a FastAPI app the frontend dashboard talks to.

Endpoints:
  POST /complaint                     -> submit a new complaint, get full risk analysis
  GET  /zones                         -> all zones with current aggregate risk (for heatmap)
  GET  /complaint/{id}/graph          -> transaction chain graph (for visualization)
  GET  /complaint/{id}/explanation    -> plain-English reasoning for that complaint
  GET  /demo/complaints                -> list seeded demo complaints (for the demo script)
  POST /demo/seed/{complaint_id}       -> replay an existing dataset complaint as if newly filed

Run with:
  uvicorn main:app --reload --port 8000
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crime_graph import CrimeGraph, load_data as load_graph_data
from risk_model import predict_zone_risks
from explainability import generate_explanation

DATA_DIR = str(Path(__file__).resolve().parent.parent / "data")
MODEL_PATH = str(Path(__file__).resolve().parent / "risk_model.joblib")

app = FastAPI(title="Cash-Out Risk Forecasting API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load everything ONCE at startup (fast, in-memory - fine for a demo)
# ---------------------------------------------------------------------------
_complaints, _chains_list, _historical, _networks = load_graph_data()
CRIME_GRAPH = CrimeGraph(_complaints, _chains_list, _historical, _networks)

with open(f"{DATA_DIR}/zones.json") as f:
    ZONES = json.load(f)
ZONES_BY_ID = {z["zone_id"]: z for z in ZONES}

_bundle = joblib.load(MODEL_PATH)
CLF = _bundle["model"]
FEATURE_COLS = _bundle["feature_cols"]
ZONE_STATS = _bundle["zone_stats"]
FEATURE_IMPORTANCE = dict(zip(FEATURE_COLS, CLF.feature_importances_))

CHAINS_BY_ID = {c["complaint_id"]: c["chain"] for c in _chains_list}
ALL_COMPLAINTS = list(_complaints)  # will grow as new complaints are submitted

# In-memory store of "active" complaints + their computed analysis, for the /zones heatmap
ACTIVE_ANALYSES = {}  # complaint_id -> analysis dict


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------
class ChainHop(BaseModel):
    hop_number: int
    from_account: str
    to_account: str
    bank: str
    amount_transferred: float
    timestamp: Optional[str] = None


class ComplaintIn(BaseModel):
    complaint_id: Optional[str] = None
    filed_time: Optional[str] = None
    victim_zone: str
    amount: float
    source_account: str
    chain: List[ChainHop] = []


class ZoneRisk(BaseModel):
    zone_id: str
    risk_score: float
    predicted_time_window: str


class ComplaintAnalysis(BaseModel):
    complaint_id: str
    match_type: str
    network_id: Optional[str]
    confidence: float
    zone_risks: List[ZoneRisk]
    explanation_summary: str
    explanation_reasons: List[str]


# ---------------------------------------------------------------------------
# Core analysis pipeline (used by both /complaint and /demo/seed)
# ---------------------------------------------------------------------------
def analyze_complaint(complaint: dict, chain: list) -> dict:
    accounts = {complaint["source_account"]} | {h["to_account"] for h in chain}
    banks = {h["bank"] for h in chain}

    match_result = CRIME_GRAPH.match_network(accounts, banks)

    if match_result and match_result.get("matched"):
        zone_history = CRIME_GRAPH.zone_history_for_network(match_result["network_id"])
        hour_range = match_result.get("favorite_hour_range") or (12, 15)
        zone_risks = [
            {
                "zone_id": zid,
                "risk_score": round(info["probability"] * 100, 1),
                "predicted_time_window": f"{hour_range[0]:02d}:00-{hour_range[1]:02d}:00",
            }
            for zid, info in zone_history.items()
        ]
        zone_risks.sort(key=lambda r: -r["risk_score"])
        ml_predictions = None
    else:
        zone_risks = predict_zone_risks(complaint, chain, ZONES, CLF, FEATURE_COLS, ZONE_STATS, top_k=5)
        ml_predictions = zone_risks

    explanation = generate_explanation(
        complaint, match_result, ml_predictions, FEATURE_IMPORTANCE, CRIME_GRAPH, ALL_COMPLAINTS
    )

    return {
        "complaint_id": complaint["complaint_id"],
        "match_type": explanation["match_type"],
        "network_id": explanation.get("network_id"),
        "confidence": explanation["confidence"],
        "zone_risks": zone_risks[:5],
        "explanation_summary": explanation["summary"],
        "explanation_reasons": explanation["reasons"],
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/complaint", response_model=ComplaintAnalysis)
def submit_complaint(complaint_in: ComplaintIn):
    complaint_id = complaint_in.complaint_id or f"CMP{len(ALL_COMPLAINTS)+1:04d}"
    filed_time = complaint_in.filed_time or datetime.now().isoformat()

    complaint = {
        "complaint_id": complaint_id,
        "filed_time": filed_time,
        "victim_zone": complaint_in.victim_zone,
        "amount": complaint_in.amount,
        "source_account": complaint_in.source_account,
        "network_id": None,  # unknown at submission time - that's what we're solving for
    }
    chain = [h.dict() for h in complaint_in.chain]

    analysis = analyze_complaint(complaint, chain)

    # Backfill the matched network onto this complaint record so FUTURE submissions
    # in this session correctly count it as "related" - this is what makes the
    # live-escalation demo (risk case count climbing as new complaints come in) work.
    complaint["network_id"] = analysis.get("network_id")

    ALL_COMPLAINTS.append(complaint)
    CHAINS_BY_ID[complaint_id] = chain
    ACTIVE_ANALYSES[complaint_id] = analysis

    return analysis


@app.get("/zones")
def get_zones():
    """
    Returns every zone with an aggregate 'current risk' for the heatmap.
    Aggregate = the highest risk score any ACTIVE complaint has assigned to that zone.
    Zones with no active complaints pointing at them show a low baseline risk.
    """
    zone_current_risk = {zid: 0.0 for zid in ZONES_BY_ID}
    zone_related_cases = {zid: 0 for zid in ZONES_BY_ID}

    for analysis in ACTIVE_ANALYSES.values():
        for zr in analysis["zone_risks"]:
            zid = zr["zone_id"]
            if zid in zone_current_risk:
                zone_current_risk[zid] = max(zone_current_risk[zid], zr["risk_score"])
                zone_related_cases[zid] += 1

    result = []
    for zid, meta in ZONES_BY_ID.items():
        result.append({
            **meta,
            "current_risk": round(zone_current_risk[zid], 1),
            "related_active_cases": zone_related_cases[zid],
        })
    return sorted(result, key=lambda z: -z["current_risk"])


@app.get("/complaint/{complaint_id}/graph")
def get_complaint_graph(complaint_id: str):
    if complaint_id not in CHAINS_BY_ID:
        raise HTTPException(status_code=404, detail="Complaint not found")
    return CRIME_GRAPH.chain_summary(complaint_id) if complaint_id in CRIME_GRAPH.chains_by_complaint \
        else {
            "nodes": [{"id": h["from_account"], "role": "account"} for h in CHAINS_BY_ID[complaint_id][:1]]
                    + [{"id": h["to_account"], "role": f"hop_{h['hop_number']}", "bank": h["bank"]} for h in CHAINS_BY_ID[complaint_id]],
            "edges": [{"from": h["from_account"], "to": h["to_account"], "amount": h["amount_transferred"], "bank": h["bank"]} for h in CHAINS_BY_ID[complaint_id]],
        }


@app.get("/complaint/{complaint_id}/explanation")
def get_complaint_explanation(complaint_id: str):
    if complaint_id in ACTIVE_ANALYSES:
        a = ACTIVE_ANALYSES[complaint_id]
        return {
            "summary": a["explanation_summary"],
            "reasons": a["explanation_reasons"],
        }
    raise HTTPException(status_code=404, detail="No analysis found for this complaint. Submit it via POST /complaint first.")


@app.get("/demo/complaints")
def list_demo_complaints():
    """Handy list for picking demo scenarios: known-network cases vs novel cases."""
    known = [c["complaint_id"] for c in _complaints if c.get("network_id")][:5]
    novel = [c["complaint_id"] for c in _complaints if not c.get("network_id")][:5]
    return {"known_network_examples": known, "novel_examples": novel}


@app.post("/demo/seed/{complaint_id}")
def seed_demo_complaint(complaint_id: str):
    """Replay an existing dataset complaint through the live pipeline, as if just filed."""
    complaint = next((c for c in _complaints if c["complaint_id"] == complaint_id), None)
    if not complaint:
        raise HTTPException(status_code=404, detail="Unknown demo complaint_id")
    chain = CHAINS_BY_ID.get(complaint_id, [])

    analysis = analyze_complaint(complaint, chain)
    ACTIVE_ANALYSES[complaint_id] = analysis
    return analysis


@app.get("/")
def root():
    return {"status": "ok", "message": "Cash-Out Risk Forecasting API is running"}
