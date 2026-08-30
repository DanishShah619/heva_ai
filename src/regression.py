

import json
import os
from datetime import datetime, timezone


def snapshot(results, path="results/baseline_snapshot.json", metadata=None):
    """Saves a minimal pass/fail snapshot keyed by question id, plus metadata
    describing what config produced it (so you know what you're diffing
    against later). Call this once, right after a run you trust."""
    snap = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
        "results_by_id": {
            r["id"]: {
                "passed": r["passed"],
                "verdict": r["verdict"],
                "hallucinated": r.get("hallucination") is not None,
            }
            for r in results
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2)
    print(f"Snapshot of {len(results)} results saved to {path}")
    return snap


def diff_against_snapshot(new_results, snapshot_path="results/baseline_snapshot.json"):
    """Compares new_results against a saved snapshot. Returns a dict with:
      - regressions: previously passed, now fails
      - new_passes: previously failed, now passes (worth knowing too — a
        prompt change can fix some things while breaking others; you want
        to see both, not just the bad news)
      - new_hallucinations: previously did not hallucinate, now does
      - resolved_hallucinations: previously hallucinated, now does not
      - missing_ids / new_ids: questions present in one run but not the other
        (e.g. if the ground truth set itself changed between runs)
    """
    with open(snapshot_path, "r", encoding="utf-8") as f:
        snap = json.load(f)
    old = snap["results_by_id"]
    new = {r["id"]: r for r in new_results}

    old_ids, new_ids_set = set(old.keys()), set(new.keys())
    common_ids = old_ids & new_ids_set

    regressions, new_passes = [], []
    new_hallucinations, resolved_hallucinations = [], []

    for qid in common_ids:
        old_row, new_row = old[qid], new[qid]
        new_hallucinated = new_row.get("hallucination") is not None

        if old_row["passed"] and not new_row["passed"]:
            regressions.append(qid)
        elif not old_row["passed"] and new_row["passed"]:
            new_passes.append(qid)

        if not old_row["hallucinated"] and new_hallucinated:
            new_hallucinations.append(qid)
        elif old_row["hallucinated"] and not new_hallucinated:
            resolved_hallucinations.append(qid)

    diff = {
        "compared_against_snapshot_created_at": snap["created_at"],
        "compared_against_metadata": snap["metadata"],
        "n_common_ids": len(common_ids),
        "regressions": regressions,
        "n_regressions": len(regressions),
        "new_passes": new_passes,
        "n_new_passes": len(new_passes),
        "new_hallucinations": new_hallucinations,
        "n_new_hallucinations": len(new_hallucinations),
        "resolved_hallucinations": resolved_hallucinations,
        "n_resolved_hallucinations": len(resolved_hallucinations),
        "missing_ids": sorted(old_ids - new_ids_set),   # in old run, not new
        "new_ids": sorted(new_ids_set - old_ids),        # in new run, not old
    }
    return diff


def print_diff_report(diff):
    print("\n=== REGRESSION REPORT ===")
    print(f"Compared against snapshot from: {diff['compared_against_snapshot_created_at']}")
    print(f"Snapshot metadata: {diff['compared_against_metadata']}")
    print(f"Questions compared: {diff['n_common_ids']}")

    if diff["n_regressions"]:
        print(f"\n⚠️  {diff['n_regressions']} REGRESSION(S) — previously passing, now failing:")
        for qid in diff["regressions"]:
            print(f"   - {qid}")
    else:
        print("\n✅ No regressions.")

    if diff["n_new_hallucinations"]:
        print(f"\n⚠️  {diff['n_new_hallucinations']} NEW HALLUCINATION(S):")
        for qid in diff["new_hallucinations"]:
            print(f"   - {qid}")

    if diff["n_new_passes"]:
        print(f"\n✅ {diff['n_new_passes']} newly passing (previously failed):")
        for qid in diff["new_passes"]:
            print(f"   - {qid}")

    if diff["n_resolved_hallucinations"]:
        print(f"\n✅ {diff['n_resolved_hallucinations']} hallucination(s) resolved:")
        for qid in diff["resolved_hallucinations"]:
            print(f"   - {qid}")

    if diff["missing_ids"] or diff["new_ids"]:
        print(f"\nNote: question set changed between runs — "
              f"{len(diff['missing_ids'])} missing, {len(diff['new_ids'])} new. "
              f"These were excluded from the comparison above.")


def run_regression_check(
    new_results_path="results/baseline.json",
    snapshot_path="results/baseline_snapshot.json",
    output_path="results/regression_diff.json",
):
    """Convenience wrapper: loads a results file, diffs it against the
    snapshot, prints and saves the report."""
    with open(new_results_path, "r", encoding="utf-8") as f:
        new_results = json.load(f)["results"]

    diff = diff_against_snapshot(new_results, snapshot_path)
    print_diff_report(diff)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(diff, f, indent=2)
    print(f"\nFull diff written to {output_path}")

    return diff


if __name__ == "__main__":
    run_regression_check()