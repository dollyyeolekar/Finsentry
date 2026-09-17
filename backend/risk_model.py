"""
Risk Scoring Model
====================
Trains a model that, given features of a NEW complaint (and its related cluster of
complaints), predicts a risk score for EACH zone (0-100) plus a likely time window.

Approach for a 48h hackathon demo (kept deliberately simple + explainable):
  - For every (complaint, candidate_zone) pair in training data, build features that
    describe "how likely is THIS zone to be the cash-out point for THIS complaint".
  - Label = 1 if that zone was the actual withdrawal zone, else 0.
  - Train a RandomForestClassifier to output a probability per zone.
  - At prediction time, score every zone for a new complaint, rank them, and also
    estimate a time window from the historical hour distribution of top zones.

This keeps everything visible/tunable in one file - good for demo debugging.
"""

import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

DATA_DIR = str(Path(__file__).resolve().parent.parent / "data")
MODEL_PATH = str(Path(__file__).resolve().parent / "risk_model.joblib")

random.seed(42)


def load_data():
    with open(f"{DATA_DIR}/complaints.json") as f:
        complaints = json.load(f)
    with open(f"{DATA_DIR}/transaction_chains.json") as f:
        chains = {c["complaint_id"]: c["chain"] for c in json.load(f)}
    with open(f"{DATA_DIR}/zones.json") as f:
        zones = json.load(f)
    with open(f"{DATA_DIR}/historical_withdrawals.json") as f:
        historical = json.load(f)
    return complaints, chains, zones, historical


def build_zone_stats(historical):
    """Precompute per-zone historical stats used as features (how 'hot' is each zone overall)."""
    zone_counts = Counter(h["zone_id"] for h in historical)
    zone_hour_sum = defaultdict(int)
    zone_hour_n = defaultdict(int)
    for h in historical:
        hour = datetime.fromisoformat(h["timestamp"]).hour
        zone_hour_sum[h["zone_id"]] += hour
        zone_hour_n[h["zone_id"]] += 1
    total = sum(zone_counts.values()) or 1
    stats = {}
    for zid, count in zone_counts.items():
        avg_hour = zone_hour_sum[zid] / max(zone_hour_n[zid], 1)
        stats[zid] = {
            "historical_withdrawal_count": count,
            "historical_share": count / total,
            "avg_historical_hour": avg_hour,
        }
    return stats


def zone_distance(zone_a, zone_b, zones_by_id):
    a, b = zones_by_id[zone_a], zones_by_id[zone_b]
    return ((a["row"] - b["row"]) ** 2 + (a["col"] - b["col"]) ** 2) ** 0.5


def build_features(complaint, chain, zone_id, zones_by_id, zone_stats):
    """
    Build one feature row describing: 'how likely is zone_id to be the cash-out
    point for this complaint'.
    """
    hop_count = len(chain)
    total_amount = sum(h["amount_transferred"] for h in chain) if chain else complaint["amount"]
    filed_hour = datetime.fromisoformat(complaint["filed_time"]).hour
    dist_from_victim = zone_distance(complaint["victim_zone"], zone_id, zones_by_id)

    zstat = zone_stats.get(zone_id, {
        "historical_withdrawal_count": 0,
        "historical_share": 0.0,
        "avg_historical_hour": 12.0,
    })

    zone_meta = zones_by_id[zone_id]

    return {
        "hop_count": hop_count,
        "total_amount": total_amount,
        "filed_hour": filed_hour,
        "dist_from_victim_zone": dist_from_victim,
        "zone_atm_density": zone_meta["atm_density"],
        "zone_historical_count": zstat["historical_withdrawal_count"],
        "zone_historical_share": zstat["historical_share"],
        "zone_avg_historical_hour": zstat["avg_historical_hour"],
        "same_zone_as_victim": int(complaint["victim_zone"] == zone_id),
    }


def build_training_table(complaints, chains, zones, zone_stats):
    zones_by_id = {z["zone_id"]: z for z in zones}
    zone_ids = list(zones_by_id.keys())
    rows = []

    for c in complaints:
        chain = chains.get(c["complaint_id"], [])
        actual_zone = c["actual_withdrawal_zone"]

        # Positive example: the real withdrawal zone
        # Negative examples: a handful of random other zones (keeps dataset balanced & fast)
        negative_zones = random.sample([z for z in zone_ids if z != actual_zone], k=4)

        for zid in [actual_zone] + negative_zones:
            feats = build_features(c, chain, zid, zones_by_id, zone_stats)
            feats["label"] = int(zid == actual_zone)
            feats["complaint_id"] = c["complaint_id"]
            feats["zone_id"] = zid
            rows.append(feats)

    return pd.DataFrame(rows)


def train_model():
    complaints, chains, zones, historical = load_data()
    zone_stats = build_zone_stats(historical)
    df = build_training_table(complaints, chains, zones, zone_stats)

    feature_cols = [
        "hop_count", "total_amount", "filed_hour", "dist_from_victim_zone",
        "zone_atm_density", "zone_historical_count", "zone_historical_share",
        "zone_avg_historical_hour", "same_zone_as_victim",
    ]

    # Split by COMPLAINT, not by row, so all candidate zones for one complaint
    # stay together (otherwise the model could "peek" at related rows)
    complaint_ids = df["complaint_id"].unique()
    train_ids, test_ids = train_test_split(complaint_ids, test_size=0.25, random_state=42)
    train_df = df[df["complaint_id"].isin(train_ids)]
    test_df = df[df["complaint_id"].isin(test_ids)]

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=3, random_state=42
    )
    clf.fit(train_df[feature_cols], train_df["label"])

    # --- Proper evaluation for a RANKING task: hit-rate@k ---
    # For each held-out complaint, does the true zone land in our top-3 predictions?
    hits_at_1, hits_at_3, total = 0, 0, 0
    for cid in test_ids:
        rows = test_df[test_df["complaint_id"] == cid].copy()
        rows["pred_prob"] = clf.predict_proba(rows[feature_cols])[:, 1]
        rows = rows.sort_values("pred_prob", ascending=False)
        true_zone_rank = rows.reset_index(drop=True)
        rank_of_true = true_zone_rank.index[true_zone_rank["label"] == 1].tolist()
        if rank_of_true:
            total += 1
            if rank_of_true[0] == 0:
                hits_at_1 += 1
            if rank_of_true[0] < 3:
                hits_at_3 += 1

    print("=== Model evaluation (ranking task: is the true zone in our top guesses?) ===")
    print(f"  Held-out complaints evaluated : {total}")
    print(f"  True zone = our #1 guess      : {hits_at_1}/{total}  ({hits_at_1/total*100:.0f}%)")
    print(f"  True zone in our top-3 guesses: {hits_at_3}/{total}  ({hits_at_3/total*100:.0f}%)")
    print("  (baseline random guess among 5 candidates per complaint would score ~20% / ~60%)")

    print("\n=== Feature importance (what the model actually relies on) ===")
    importance = sorted(
        zip(feature_cols, clf.feature_importances_), key=lambda x: -x[1]
    )
    for name, score in importance:
        print(f"  {name:28s} {score:.3f}")

    joblib.dump({
        "model": clf,
        "feature_cols": feature_cols,
        "zone_stats": zone_stats,
    }, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    return clf, feature_cols, zone_stats


def predict_zone_risks(complaint, chain, zones, clf, feature_cols, zone_stats, top_k=5):
    """
    Given a (possibly new/unseen) complaint + its chain, score every zone and
    return the top_k most likely cash-out zones with risk scores (0-100) and
    an estimated time window.
    """
    zones_by_id = {z["zone_id"]: z for z in zones}
    rows = []
    for zid in zones_by_id:
        feats = build_features(complaint, chain, zid, zones_by_id, zone_stats)
        rows.append({"zone_id": zid, **feats})

    df = pd.DataFrame(rows)
    probs = clf.predict_proba(df[feature_cols])[:, 1]  # probability of class 1 (this zone)
    df["risk_score"] = (probs * 100).round(1)

    # normalize so top zones look like the "Zone B: 78%, Zone C: 61%" style output
    df = df.sort_values("risk_score", ascending=False).head(top_k)

    results = []
    for _, row in df.iterrows():
        zstat = zone_stats.get(row["zone_id"], {"avg_historical_hour": 12})
        avg_hour = int(round(zstat.get("avg_historical_hour", 12)))
        window_start = max(0, avg_hour - 1)
        window_end = min(23, avg_hour + 2)
        results.append({
            "zone_id": row["zone_id"],
            "risk_score": float(row["risk_score"]),
            "predicted_time_window": f"{window_start:02d}:00-{window_end:02d}:00",
        })
    return results


if __name__ == "__main__":
    clf, feature_cols, zone_stats = train_model()

    # Quick sanity check: predict on a NOVEL complaint (no network match) to see
    # the model working stand-alone, without the network-matcher shortcut.
    complaints, chains, zones, historical = load_data()
    novel_complaints = [c for c in complaints if c["network_id"] is None]

    if novel_complaints:
        test_c = novel_complaints[0]
        test_chain = chains.get(test_c["complaint_id"], [])
        print(f"\n=== Predicting for NOVEL complaint {test_c['complaint_id']} ===")
        print(f"(true actual withdrawal zone was: {test_c['actual_withdrawal_zone']})")
        preds = predict_zone_risks(test_c, test_chain, zones, clf, feature_cols, zone_stats)
        for p in preds:
            print(f"  {p['zone_id']}: risk {p['risk_score']}/100, window {p['predicted_time_window']}")
