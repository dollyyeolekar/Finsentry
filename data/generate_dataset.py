"""
Synthetic Dataset Generator
============================
Creates a fake but realistic dataset for the Cash-Out Risk Landscape demo:
  - A 5x5 city zone grid (like Pune North/South/East/West/Center)
  - Cybercrime complaints
  - Mule account transaction chains (2-4 hops)
  - Historical withdrawal records, some linked into repeating "networks"

Output files (all in data/):
  - zones.json
  - complaints.json
  - transaction_chains.json
  - historical_withdrawals.json
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)  # reproducible demo data

OUTPUT_DIR = str(Path(__file__).resolve().parent)

# ---------------------------------------------------------------------------
# 1. Build the zone grid (5x5 = 25 zones)
# ---------------------------------------------------------------------------
ZONE_ROWS = 5
ZONE_COLS = 5
BANKS = ["SBI", "HDFC", "ICICI", "Axis", "PNB", "Kotak", "BOB"]

def build_zones():
    zones = []
    row_labels = ["North", "North-Center", "Center", "South-Center", "South"]
    col_labels = ["West", "West-Center", "Center", "East-Center", "East"]
    zid = 1
    for r in range(ZONE_ROWS):
        for c in range(ZONE_COLS):
            zones.append({
                "zone_id": f"Z{zid}",
                "row": r,
                "col": c,
                "label": f"{row_labels[r]} / {col_labels[c]}",
                # fake lat/long so the map still has something to plot
                # centered loosely around Pune (18.52, 73.85)
                "lat": round(18.42 + r * 0.05, 4),
                "lng": round(73.75 + c * 0.05, 4),
                "atm_density": random.randint(3, 15)  # ATMs in this zone
            })
            zid += 1
    return zones

ZONES = build_zones()
ZONE_IDS = [z["zone_id"] for z in ZONES]

# ---------------------------------------------------------------------------
# 2. Define a handful of "known mule networks" that reuse the same zones
#    (this is what the Network Matcher in Step 2 will detect)
# ---------------------------------------------------------------------------
NUM_NETWORKS = 6
NETWORKS = []
for i in range(NUM_NETWORKS):
    fav_zones = random.sample(ZONE_IDS, k=random.randint(2, 3))
    NETWORKS.append({
        "network_id": f"NET{i+1}",
        "favorite_zones": fav_zones,
        "favorite_hour_range": (random.randint(9, 15), random.randint(16, 22)),
        "typical_hop_count": random.randint(2, 4),
    })

def random_account_id():
    return "ACC" + "".join(random.choices("0123456789", k=8))

def random_ifsc(bank):
    return bank[:4].upper() + "0" + "".join(random.choices("0123456789", k=6))

# ---------------------------------------------------------------------------
# 3. Generate complaints + transaction chains
# ---------------------------------------------------------------------------
NUM_COMPLAINTS = 80
BASE_TIME = datetime(2026, 8, 1, 8, 0, 0)

complaints = []
transaction_chains = []

for i in range(1, NUM_COMPLAINTS + 1):
    complaint_id = f"CMP{i:04d}"
    filed_time = BASE_TIME + timedelta(
        days=random.randint(0, 20),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59)
    )
    amount = random.choice([5000, 8000, 15000, 25000, 42000, 60000, 90000, 150000])
    victim_zone = random.choice(ZONE_IDS)

    # 70% of complaints belong to a known network (repeat pattern),
    # 30% are "novel" so the ML model has to work for those
    is_known_network = random.random() < 0.7
    network = random.choice(NETWORKS) if is_known_network else None

    hop_count = network["typical_hop_count"] if network else random.randint(2, 4)

    # Build the mule chain
    chain = []
    source_account = random_account_id()
    prev_account = source_account
    for hop in range(1, hop_count + 1):
        bank = random.choice(BANKS)
        next_account = random_account_id()
        chain.append({
            "hop_number": hop,
            "from_account": prev_account,
            "to_account": next_account,
            "bank": bank,
            "ifsc": random_ifsc(bank),
            "amount_transferred": round(amount * random.uniform(0.85, 1.0)),
            "timestamp": (filed_time + timedelta(minutes=hop * random.randint(10, 90))).isoformat()
        })
        prev_account = next_account

    # Decide the withdrawal zone: if known network, bias toward its favorite zones
    if network:
        withdrawal_zone = random.choice(network["favorite_zones"])
        wd_hour = random.randint(*network["favorite_hour_range"])
    else:
        # Novel / unmatched cases still follow a REAL (but subtler) pattern instead of
        # pure randomness: higher-value frauds tend to cash out further from the victim
        # (more layering to cover distance) and in zones with higher ATM density
        # (easier to blend in). This gives the ML model genuine signal to learn from,
        # while still being a "new" pattern with no known network match.
        victim = next(z for z in ZONES if z["zone_id"] == victim_zone)
        weights = []
        for z in ZONES:
            dist = ((z["row"] - victim["row"]) ** 2 + (z["col"] - victim["col"]) ** 2) ** 0.5
            distance_pull = dist if amount > 40000 else max(0.5, 3 - dist)  # far if big fraud, near if small
            density_pull = z["atm_density"]
            weights.append(max(0.1, distance_pull) * density_pull)
        withdrawal_zone = random.choices(ZONE_IDS, weights=weights, k=1)[0]
        # time also loosely tied to amount: bigger frauds cashed out later in the day
        wd_hour = random.randint(17, 23) if amount > 40000 else random.randint(8, 16)

    withdrawal_time = chain[-1]["timestamp"]
    withdrawal_dt = datetime.fromisoformat(withdrawal_time).replace(hour=wd_hour % 24)

    complaints.append({
        "complaint_id": complaint_id,
        "filed_time": filed_time.isoformat(),
        "victim_zone": victim_zone,
        "amount": amount,
        "source_account": source_account,
        "network_id": network["network_id"] if network else None,  # ground truth (hidden from model at prediction time)
        "final_account": prev_account,
        "actual_withdrawal_zone": withdrawal_zone,       # ground truth label
        "actual_withdrawal_time": withdrawal_dt.isoformat(),
        "hop_count": hop_count
    })

    transaction_chains.append({
        "complaint_id": complaint_id,
        "chain": chain
    })

# ---------------------------------------------------------------------------
# 4. Generate historical withdrawal records (older cases, used for pattern learning)
#    These simulate "cases already solved" that show each network's habits
# ---------------------------------------------------------------------------
NUM_HISTORICAL = 150
historical_withdrawals = []
for i in range(1, NUM_HISTORICAL + 1):
    network = random.choice(NETWORKS)
    zone = random.choice(network["favorite_zones"])
    hour = random.randint(*network["favorite_hour_range"])
    ts = BASE_TIME - timedelta(days=random.randint(30, 180), hours=-hour)
    historical_withdrawals.append({
        "record_id": f"HW{i:04d}",
        "network_id": network["network_id"],
        "zone_id": zone,
        "timestamp": ts.isoformat(),
        "amount": random.choice([5000, 10000, 20000, 40000, 70000]),
        "linked_account_cluster_id": network["network_id"]
    })

# ---------------------------------------------------------------------------
# 5. Save everything
# ---------------------------------------------------------------------------
def save(name, obj):
    path = f"{OUTPUT_DIR}/{name}"
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved {path}  ({len(obj)} records)")

save("zones.json", ZONES)
save("complaints.json", complaints)
save("transaction_chains.json", transaction_chains)
save("historical_withdrawals.json", historical_withdrawals)

with open(f"{OUTPUT_DIR}/networks.json", "w") as f:
    json.dump(NETWORKS, f, indent=2)
print(f"Saved {OUTPUT_DIR}/networks.json  ({len(NETWORKS)} records)")

print("\nDataset generation complete.")
print(f"Zones: {len(ZONES)} | Complaints: {len(complaints)} | Historical withdrawals: {len(historical_withdrawals)} | Networks: {len(NETWORKS)}")
