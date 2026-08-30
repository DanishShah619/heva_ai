
import json

import matplotlib.pyplot as plt
import numpy as np


def load_results(*paths):
    """Loads and concatenates the 'results' list from one or more results
    JSON files (e.g. baseline.json and adversarial_results.json), so
    calibration can be assessed across clean + adversarial inputs together,
    which is what the assignment actually asks for — not just the clean set."""
    all_results = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_results.extend(data["results"])
    return all_results


def compute_calibration(results, bin_edges=(0.0, 0.5, 0.7, 0.85, 1.01)):
    """Buckets results by model_confidence, computes actual accuracy per
    bucket, and Expected Calibration Error (ECE) = the weighted average gap
    between confidence and accuracy across buckets.

    Rows with confidence=None are excluded and counted separately — a model
    that frequently omits confidence is itself worth reporting."""
    usable = [r for r in results if r.get("model_confidence") is not None]
    n_missing_confidence = len(results) - len(usable)

    buckets = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        bucket_rows = [r for r in usable if lo <= r["model_confidence"] < hi]
        if not bucket_rows:
            buckets.append({
                "range": f"[{lo:.2f}, {hi:.2f})", "n": 0,
                "avg_confidence": None, "accuracy": None,
            })
            continue
        avg_conf = np.mean([r["model_confidence"] for r in bucket_rows])
        acc = np.mean([r["passed"] for r in bucket_rows])
        buckets.append({
            "range": f"[{lo:.2f}, {hi:.2f})",
            "n": len(bucket_rows),
            "avg_confidence": round(float(avg_conf), 4),
            "accuracy": round(float(acc), 4),
        })

    populated = [b for b in buckets if b["n"] > 0]
    total_n = sum(b["n"] for b in populated)
    ece = sum(
        b["n"] * abs(b["accuracy"] - b["avg_confidence"]) for b in populated
    ) / total_n if total_n else None

    return {
        "buckets": buckets,
        "ece": round(ece, 4) if ece is not None else None,
        "n_missing_confidence": n_missing_confidence,
        "n_used": len(usable),
    }


def plot_calibration_curve(calibration_result, output_path="results/calibration_curve.png",
                            title="Confidence Calibration"):
    populated = [b for b in calibration_result["buckets"] if b["n"] > 0]
    if not populated:
        print("No populated buckets — nothing to plot.")
        return

    conf = [b["avg_confidence"] for b in populated]
    acc = [b["accuracy"] for b in populated]
    sizes = [b["n"] for b in populated]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.scatter(conf, acc, s=[n * 20 for n in sizes], alpha=0.7, color="tab:blue",
               label="Observed (bubble size = n questions)")
    for c, a, n in zip(conf, acc, sizes):
        ax.annotate(f"n={n}", (c, a), textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xlabel("Model self-reported confidence")
    ax.set_ylabel("Actual accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{title}\nECE = {calibration_result['ece']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Calibration curve saved to {output_path}")
    plt.show()


def run_calibration_analysis(
    baseline_path="results/baseline.json",
    adversarial_path="results/adversarial_results.json",
    output_json_path="results/calibration.json",
    output_plot_path="results/calibration_curve.png",
    include_adversarial=True,
):
    paths = [baseline_path]
    if include_adversarial:
        import os
        if os.path.exists(adversarial_path):
            paths.append(adversarial_path)
        else:
            print(f"Note: {adversarial_path} not found — calibrating on baseline only.")

    results = load_results(*paths)
    calibration_result = compute_calibration(results)

    print("\n=== CALIBRATION SUMMARY ===")
    print(json.dumps(calibration_result, indent=2))

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(calibration_result, f, indent=2)

    plot_calibration_curve(calibration_result, output_plot_path)

    return calibration_result


if __name__ == "__main__":
    run_calibration_analysis()