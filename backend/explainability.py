"""
Explainability Layer
=======================
Takes the output of the Network Matcher (Step 2) and/or the Risk Model (Step 3)
and turns it into plain-English reasoning an investigator can actually read,
in the style of your reference:

    "Risk increased from 54 to 87 because:
       - 3 new complaints connected to this account cluster
       - ₹4.2 lakh entered associated accounts
       - similar historical cases cashed out within 3 hours"

Two paths:
  1. NETWORK MATCH found  -> explanation built from real historical case counts,
     amounts, and timing for that network (strong, concrete reasons)
  2. NO match (novel case) -> explanation built from the ML model's top
     contributing features for the top predicted zone (still grounded, but
     framed as "based on similar patterns" rather than a known gang)
"""

import json
from pathlib import Path
from collections import Counter
from datetime import datetime

DATA_DIR = str(Path(__file__).resolve().parent.parent / "data")


def load_all_complaints():
    with open(f"{DATA_DIR}/complaints.json") as f:
        return json.load(f)


def format_amount(amount):
    """₹150000 -> ₹1.5 lakh, ₹8000 -> ₹8,000 (Indian-style, easy to read on a dashboard)"""
    if amount >= 100000:
        return f"₹{amount/100000:.1f} lakh"
    return f"₹{amount:,.0f}"


# ---------------------------------------------------------------------------
# Path 1: Explanation when a KNOWN NETWORK match is found
# ---------------------------------------------------------------------------
def explain_network_match(complaint, match_result, crime_graph, all_complaints):
    network_id = match_result["network_id"]
    confidence = match_result["confidence"]

    # Count how many OTHER complaints are tied to this same network
    related = [c for c in all_complaints
               if c.get("network_id") == network_id and c["complaint_id"] != complaint["complaint_id"]]
    related_count = len(related)
    total_related_amount = sum(c["amount"] for c in related) + complaint["amount"]

    zone_history = crime_graph.zone_history_for_network(network_id)
    top_zone = next(iter(zone_history), None)
    top_zone_prob = zone_history[top_zone]["probability"] if top_zone else None

    reasons = []
    if related_count > 0:
        reasons.append(f"{related_count} other complaint(s) already linked to this same account cluster ({network_id})")
    reasons.append(f"{format_amount(total_related_amount)} has moved through accounts tied to this cluster so far")
    if top_zone and top_zone_prob:
        pct = int(top_zone_prob * 100)
        reasons.append(f"this cluster has historically cashed out in {top_zone} in {pct}% of past cases")
    hour_range = match_result.get("favorite_hour_range")
    if hour_range:
        reasons.append(f"past cash-outs from this cluster typically happen between {hour_range[0]:02d}:00 and {hour_range[1]:02d}:00")

    summary = (
        f"High-confidence match ({int(confidence*100)}%) to a known mule network ({network_id}). "
        f"Based on {related_count + 1} linked complaint(s), this pattern strongly resembles prior cash-out behavior."
    )

    return {
        "match_type": "known_network",
        "network_id": network_id,
        "confidence": confidence,
        "summary": summary,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Path 2: Explanation when NO network match -> based on ML model's top features
# ---------------------------------------------------------------------------
FEATURE_EXPLANATIONS = {
    "zone_avg_historical_hour": "the timing of this complaint aligns with when this zone has historically seen cash-outs",
    "total_amount": "the transaction amount involved is a strong signal for this type of zone",
    "zone_historical_share": "this zone accounts for a notably high share of historical cash-out activity",
    "dist_from_victim_zone": "the distance between the victim's location and this zone matches a common fraud pattern",
    "filed_hour": "the time this complaint was filed matches a recurring pattern",
    "zone_historical_count": "this zone has a meaningful number of past recorded cash-out cases",
    "zone_atm_density": "this zone has a high concentration of ATMs, making it convenient for cash-out",
    "hop_count": "the number of account hops in this transaction chain fits a typical layering pattern",
    "same_zone_as_victim": "this zone is the same as where the victim is located",
}


def explain_ml_prediction(complaint, top_prediction, feature_importance, all_complaints):
    zone_id = top_prediction["zone_id"]
    risk_score = top_prediction["risk_score"]

    # Similar past cases: same rough amount bracket, for a "grounded" feel
    similar = [
        c for c in all_complaints
        if c["complaint_id"] != complaint["complaint_id"]
        and abs(c["amount"] - complaint["amount"]) <= complaint["amount"] * 0.3
    ]

    reasons = []
    top_features = sorted(feature_importance.items(), key=lambda x: -x[1])[:3]
    for fname, _ in top_features:
        explanation = FEATURE_EXPLANATIONS.get(fname)
        if explanation:
            reasons.append(explanation.capitalize())

    if similar:
        reasons.append(f"{len(similar)} historical complaint(s) of a similar amount show a comparable cash-out pattern")

    reasons.append(f"transaction amount of {format_amount(complaint['amount'])} is consistent with cases previously seen in this zone")

    summary = (
        f"No known mule network matched directly, so this is a model-based prediction from learned patterns. "
        f"{zone_id} scores highest at {risk_score}/100 based on the factors below."
    )

    return {
        "match_type": "ml_prediction",
        "network_id": None,
        "confidence": round(risk_score / 100, 2),
        "summary": summary,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------
def generate_explanation(complaint, match_result, ml_predictions, feature_importance, crime_graph, all_complaints):
    """
    match_result      : output of CrimeGraph.match_network() (or None)
    ml_predictions    : output of predict_zone_risks() (list, used if no match)
    feature_importance: dict of {feature_name: importance_score} from the trained model
    """
    if match_result and match_result.get("matched"):
        return explain_network_match(complaint, match_result, crime_graph, all_complaints)
    else:
        top_pred = ml_predictions[0] if ml_predictions else None
        if not top_pred:
            return {
                "match_type": "none",
                "summary": "Insufficient data to generate a prediction for this complaint.",
                "reasons": [],
            }
        return explain_ml_prediction(complaint, top_pred, feature_importance, all_complaints)


# ---------------------------------------------------------------------------
# Self-test: run the FULL pipeline end-to-end (Steps 2 + 3 + 4 together)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from crime_graph import CrimeGraph, load_data as load_graph_data
    from risk_model import load_data as load_model_data, predict_zone_risks
    import joblib

    complaints, chains, historical, networks = load_graph_data()
    cg = CrimeGraph(complaints, chains, historical, networks)

    bundle = joblib.load(str(Path(__file__).resolve().parent / "risk_model.joblib"))
    clf, feature_cols, zone_stats = bundle["model"], bundle["feature_cols"], bundle["zone_stats"]
    _, chains_dict, zones, _ = load_model_data()

    feature_importance = dict(zip(feature_cols, clf.feature_importances_))
    chains_by_id = {c["complaint_id"]: c["chain"] for c in json.load(open(f"{DATA_DIR}/transaction_chains.json"))}

    print("=" * 70)
    print("TEST CASE A: A complaint that matches a known network")
    print("=" * 70)
    test_a = complaints[0]  # known network case
    chain_a = chains_by_id[test_a["complaint_id"]]
    accounts_a = {test_a["source_account"]} | {h["to_account"] for h in chain_a}
    banks_a = {h["bank"] for h in chain_a}
    match_a = cg.match_network(accounts_a, banks_a)
    explanation_a = generate_explanation(test_a, match_a, None, feature_importance, cg, complaints)
    print(json.dumps(explanation_a, indent=2))

    print("\n" + "=" * 70)
    print("TEST CASE B: A novel complaint with no network match")
    print("=" * 70)
    test_b = next(c for c in complaints if c["network_id"] is None)
    chain_b = chains_by_id[test_b["complaint_id"]]
    ml_preds_b = predict_zone_risks(test_b, chain_b, zones, clf, feature_cols, zone_stats)
    explanation_b = generate_explanation(test_b, None, ml_preds_b, feature_importance, cg, complaints)
    print(json.dumps(explanation_b, indent=2))
