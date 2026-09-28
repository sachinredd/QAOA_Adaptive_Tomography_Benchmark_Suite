"""Plots and exact machine-readable resource comparisons from saved runs."""
from pathlib import Path
import csv
import json
import numpy as np


def plot_benchmark(output):
    """Mean curves with sample SD bands; one seed has no uncertainty band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    config = json.loads((output/"config.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), constrained_layout=True)
    for policy in config["policies"]:
        histories = [json.loads((output/f"seed_{seed}"/policy/"history.json").read_text()) for seed in config["seeds"]]
        x = np.array([r["characterization_shots"] for r in histories[0]])
        y = np.array([[r["metrics"]["heldout_rmse"] for r in h] for h in histories])
        line, = axes[0].plot(x, y.mean(0), marker="o", ms=3, label=policy)
        if len(histories) > 1:
            sd = y.std(0, ddof=1)
            axes[0].fill_between(x, np.maximum(0., y.mean(0)-sd), y.mean(0)+sd, alpha=.12, color=line.get_color())
        all_shots = np.array([[r["total_attempted_shots"] for r in h] for h in histories])
        axes[1].plot(all_shots.mean(0), y.mean(0), marker="o", ms=3, label=policy)
        axes[2].bar(policy, np.mean([h[-1]["algorithm_seconds"] for h in histories]), color=line.get_color())
    axes[0].set(xlabel="Characterization shots", ylabel="Held-out probability RMSE", title="Equal characterization budget")
    axes[1].set(xlabel="Total attempted shots", ylabel="Held-out probability RMSE", title="Including QAOA shots", xscale="log")
    axes[2].set(ylabel="Recorded seconds", title="Acquisition plus fit plus selection")
    axes[2].tick_params(axis="x", labelrotation=35)
    axes[0].legend(fontsize=8)
    for ax in axes[:2]:
        ax.grid(alpha=.2)
    fig.savefig(output/"comparison.png", dpi=170)
    plt.close(fig)


def write_summary_csv(output):
    """One row per policy and seed, with no rounding of stored values."""
    output = Path(output)
    results = json.loads((output/"summary.json").read_text())
    rows = []
    for item in results:
        f = item["final"]
        rows.append(dict(seed=item["seed"], policy=item["policy"], **f["metrics"],
            characterization_shots=f["characterization_shots"], total_attempted_shots=f["total_attempted_shots"],
            channel_applications=f["channel_applications"], algorithm_seconds=f["algorithm_seconds"],
            optimizer_circuits=f["optimizer_circuits"], first_target_shots=item["first_target_shots"]))
    with (output/"summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
