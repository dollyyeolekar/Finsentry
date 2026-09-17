"""
Crime Graph + Network Matcher
===============================
Builds a graph connecting: complaint -> source account -> mule accounts -> withdrawal zone.
Then, given a NEW complaint's account chain, checks whether it matches a previously-seen
mule network by looking at shared accounts / shared banks+IFSC patterns.

If a match is found -> we already know that network's favorite zones and hours from
historical_withdrawals.json, so we can return an instant, high-confidence prediction.

If no match -> the complaint is "novel" and gets passed to the ML risk model (Step 3).
"""

import json
from pathlib import Path
import networkx as nx
from collections import defaultdict, Counter

DATA_DIR = str(Path(__file__).resolve().parent.parent / "data")


def load_data():
    with open(f"{DATA_DIR}/complaints.json") as f:
        complaints = json.load(f)
    with open(f"{DATA_DIR}/transaction_chains.json") as f:
        chains = json.load(f)
    with open(f"{DATA_DIR}/historical_withdrawals.json") as f:
        historical = json.load(f)
    with open(f"{DATA_DIR}/networks.json") as f:
        networks = json.load(f)
    return complaints, chains, historical, networks


class CrimeGraph:
    """
    Wraps a NetworkX directed graph of the whole crime dataset and provides
    network-matching + zone-history lookups.
    """

    def __init__(self, complaints, chains, historical, networks):
        self.G = nx.DiGraph()
        self.complaints_by_id = {c["complaint_id"]: c for c in complaints}
        self.chains_by_complaint = {c["complaint_id"]: c["chain"] for c in chains}
        self.historical = historical
        self.networks = networks

        # account -> known network_id, built from historical data + labeled training complaints
        self.account_to_network = {}

        self._build_graph(complaints, chains)
        self._learn_network_signatures(complaints, chains)

    # ------------------------------------------------------------------
    def _build_graph(self, complaints, chains):
        """Add every complaint's chain as edges: victim_zone -> source_acc -> ... -> final_acc -> withdrawal_zone"""
        for c in complaints:
            complaint_id = c["complaint_id"]
            chain = self.chains_by_complaint.get(complaint_id, [])

            self.G.add_node(f"complaint:{complaint_id}", type="complaint", **c)
            self.G.add_edge(f"zone:{c['victim_zone']}", f"complaint:{complaint_id}", relation="filed_in")
            self.G.add_edge(f"complaint:{complaint_id}", f"account:{c['source_account']}", relation="source")

            for hop in chain:
                self.G.add_edge(
                    f"account:{hop['from_account']}",
                    f"account:{hop['to_account']}",
                    relation="transfer",
                    amount=hop["amount_transferred"],
                    bank=hop["bank"],
                    ifsc=hop["ifsc"],
                    timestamp=hop["timestamp"],
                )

            if chain:
                final_account = chain[-1]["to_account"]
                self.G.add_edge(
                    f"account:{final_account}",
                    f"zone:{c['actual_withdrawal_zone']}",
                    relation="withdrawn_at",
                )

    # ------------------------------------------------------------------
    def _learn_network_signatures(self, complaints, chains):
        """
        Build a lookup: for every account that appeared in a complaint whose
        ground-truth network_id we know, remember which network it belongs to.
        In a real system this would come from investigator-confirmed cases;
        here we use our labeled training complaints as the 'known' set.
        """
        for c in complaints:
            net_id = c.get("network_id")
            if not net_id:
                continue
            chain = self.chains_by_complaint.get(c["complaint_id"], [])
            accounts = {c["source_account"]}
            for hop in chain:
                accounts.add(hop["from_account"])
                accounts.add(hop["to_account"])
            for acc in accounts:
                self.account_to_network[acc] = net_id

        # also index bank+IFSC combos per network, as a fuzzy fallback signal
        self.network_bank_signature = defaultdict(Counter)
        for c in complaints:
            net_id = c.get("network_id")
            if not net_id:
                continue
            for hop in self.chains_by_complaint.get(c["complaint_id"], []):
                self.network_bank_signature[net_id][hop["bank"]] += 1

    # ------------------------------------------------------------------
    def match_network(self, new_accounts, new_banks):
        """
        Given a NEW complaint's set of accounts (and banks used in its chain),
        try to match it to a known network.

        IMPORTANT: only a REUSED ACCOUNT counts as real evidence of a network match.
        Bank overlap alone is not used to declare a match - with only a handful of
        banks in circulation, almost any complaint would coincidentally share a bank
        with almost any network, producing false positives. Bank signature is kept
        only as a minor tie-breaker between candidates that already have a direct hit.

        Returns: dict with match info, or None if no match found.
        """
        # Direct account match is the ONLY thing that can trigger a match
        votes = Counter()
        for acc in new_accounts:
            net_id = self.account_to_network.get(acc)
            if net_id:
                votes[net_id] += 3

        if not votes:
            return None

        # Bank pattern similarity used only to break ties between already-voted networks
        for net_id in list(votes.keys()):
            bank_counter = self.network_bank_signature.get(net_id, Counter())
            overlap = sum(1 for b in new_banks if b in bank_counter)
            votes[net_id] += overlap * 0.2

        best_network_id, score = votes.most_common(1)[0]
        confidence = min(0.95, 0.5 + score / 10)  # simple confidence heuristic

        network_meta = next((n for n in self.networks if n["network_id"] == best_network_id), None)

        return {
            "matched": True,
            "network_id": best_network_id,
            "confidence": round(confidence, 2),
            "favorite_zones": network_meta["favorite_zones"] if network_meta else [],
            "favorite_hour_range": network_meta["favorite_hour_range"] if network_meta else None,
        }

    # ------------------------------------------------------------------
    def zone_history_for_network(self, network_id):
        """Return historical withdrawal zone counts for a given network (for the heatmap)."""
        records = [h for h in self.historical if h["network_id"] == network_id]
        zone_counts = Counter(r["zone_id"] for r in records)
        total = sum(zone_counts.values()) or 1
        return {
            zone: {"count": count, "probability": round(count / total, 2)}
            for zone, count in zone_counts.most_common()
        }

    # ------------------------------------------------------------------
    def chain_summary(self, complaint_id):
        """Simple node-link structure for frontend graph visualization."""
        chain = self.chains_by_complaint.get(complaint_id, [])
        complaint = self.complaints_by_id.get(complaint_id, {})
        nodes = [{"id": complaint["source_account"], "role": "source"}]
        edges = []
        for hop in chain:
            nodes.append({"id": hop["to_account"], "role": f"mule_hop_{hop['hop_number']}", "bank": hop["bank"]})
            edges.append({
                "from": hop["from_account"],
                "to": hop["to_account"],
                "amount": hop["amount_transferred"],
                "bank": hop["bank"],
                "timestamp": hop["timestamp"],
            })
        return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Quick self-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    complaints, chains, historical, networks = load_data()
    cg = CrimeGraph(complaints, chains, historical, networks)

    print(f"Graph built: {cg.G.number_of_nodes()} nodes, {cg.G.number_of_edges()} edges\n")

    # Test 1: try matching an EXISTING complaint's accounts (should match strongly)
    test_complaint = complaints[0]
    chain = cg.chains_by_complaint[test_complaint["complaint_id"]]
    test_accounts = {test_complaint["source_account"]} | {h["to_account"] for h in chain}
    test_banks = {h["bank"] for h in chain}

    print(f"Testing match for {test_complaint['complaint_id']} (true network: {test_complaint['network_id']})")
    result = cg.match_network(test_accounts, test_banks)
    print("Match result:", json.dumps(result, indent=2))

    if result and result["matched"]:
        print(f"\nHistorical zone spread for {result['network_id']}:")
        print(json.dumps(cg.zone_history_for_network(result["network_id"]), indent=2))

    print("\nChain summary for", test_complaint["complaint_id"])
    print(json.dumps(cg.chain_summary(test_complaint["complaint_id"]), indent=2))
