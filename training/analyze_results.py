
from pathlib import Path
import csv
import json
import statistics

import matplotlib.pyplot as plt

# ==========================================
# 1. PATHS AND EXPERIMENT DESIGN
# ==========================================

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "training" / "results"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2026]
PLACEMENTS = ["early", "middle", "late"]

CONFIGURATIONS = [
    "baseline",
    "se_early",
    "se_middle",
    "se_late",
    "cbam_early",
    "cbam_middle",
    "cbam_late",
]

EXPECTED_MAX_EPOCHS = 5
EXPECTED_PATIENCE = 4

# ==========================================
# 2. LOAD AND VALIDATE ALL 21 RUNS
# ==========================================

runs = []

for config in CONFIGURATIONS:
    for seed in SEEDS:
        name = f"{config}_seed{seed}"

        metadata_path = (
            RESULTS_DIR / f"{name}_metadata.json"
        )
        history_path = (
            RESULTS_DIR / f"{name}_history.csv"
        )

        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)

        if not history_path.exists():
            raise FileNotFoundError(history_path)

        with open(metadata_path) as file:
            metadata = json.load(file)

        with open(history_path, newline="") as file:
            history = list(csv.DictReader(file))

        if not history:
            raise ValueError(f"Empty history: {name}")

        assert metadata["experiment"] == name
        assert int(metadata["seed"]) == seed
        assert metadata["max_epochs"] == EXPECTED_MAX_EPOCHS
        assert metadata["patience"] == EXPECTED_PATIENCE
        assert metadata["batch_size"] == 16
        assert metadata["learning_rate"] == 1e-4
        assert metadata["checkpoint_metric"] == (
            "validation_macro_f1"
        )
        assert metadata["augmentation"] is False
        assert len(history) == metadata["epochs_completed"]

        best_row = max(
            history,
            key=lambda r: float(r["val_macro_f1"])
        )

        best_f1 = float(best_row["val_macro_f1"])
        best_epoch = int(best_row["epoch"])

        if abs(
            best_f1 - metadata["best_val_macro_f1"]
        ) > 1e-8:
            raise ValueError(
                f"Checkpoint metric mismatch: {name}"
            )

        assert best_epoch == metadata["best_epoch"]

        runs.append({
            "configuration": config,
            "seed": seed,
            "best_epoch": best_epoch,
            "val_macro_f1": best_f1,
            "val_accuracy": float(
                best_row["val_accuracy"]
            ),
            "val_loss": float(
                best_row["val_loss"]
            ),
            "epochs_completed": len(history),
            "stopped_early": metadata["stopped_early"],
            "total_seconds": metadata["total_seconds"],
        })

assert len(runs) == 21

print("Successfully validated 21 experiments.")


# ==========================================
# 3. HELPER FOR WRITING CSV
# ==========================================

def write_csv(filename, rows):
    path = ANALYSIS_DIR / filename

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", path)


# ==========================================
# 4. EXPERIMENT-LEVEL RESULTS
# ==========================================

write_csv("experiment_summary.csv", runs)


# ==========================================
# 5. CONFIGURATION MEAN AND STD
# ==========================================

summary = []

for config in CONFIGURATIONS:
    selected = [
        r for r in runs
        if r["configuration"] == config
    ]

    f1s = [r["val_macro_f1"] for r in selected]
    accuracies = [
        r["val_accuracy"] for r in selected
    ]
    runtimes = [
        r["total_seconds"] for r in selected
    ]

    summary.append({
        "configuration": config,
        "mean_macro_f1": statistics.mean(f1s),
        "std_macro_f1": statistics.stdev(f1s),
        "mean_accuracy": statistics.mean(accuracies),
        "std_accuracy": statistics.stdev(accuracies),
        "mean_runtime_seconds": statistics.mean(runtimes),
        "best_epochs": ",".join(
            str(r["best_epoch"]) for r in selected
        ),
    })

write_csv("configuration_summary.csv", summary)


# ==========================================
# 6. PLACEMENT RANKINGS PER SEED
# ==========================================

rankings = []

for mechanism in ["se", "cbam"]:
    for seed in SEEDS:
        scores = {}

        for placement in PLACEMENTS:
            config = f"{mechanism}_{placement}"

            run = next(
                r for r in runs
                if r["configuration"] == config
                and r["seed"] == seed
            )

            scores[placement] = run["val_macro_f1"]

        ordered = sorted(
            PLACEMENTS,
            key=lambda p: scores[p],
            reverse=True
        )

        rankings.append({
            "mechanism": mechanism,
            "seed": seed,
            "early_f1": scores["early"],
            "middle_f1": scores["middle"],
            "late_f1": scores["late"],
            "rank_1": ordered[0],
            "rank_2": ordered[1],
            "rank_3": ordered[2],
        })

write_csv("placement_rankings.csv", rankings)


# ==========================================
# 7. CONFIGURATION COMPARISON FIGURE
# ==========================================

labels = [
    "Baseline",
    "SE Early",
    "SE Middle",
    "SE Late",
    "CBAM Early",
    "CBAM Middle",
    "CBAM Late",
]

means = [r["mean_macro_f1"] for r in summary]
stds = [r["std_macro_f1"] for r in summary]

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(
    labels,
    means,
    yerr=stds,
    capsize=5,
    alpha=0.8
)

ax.set_ylabel("Validation Macro-F1")
ax.set_title(
    "Model Comparison — Mean Macro-F1 ± SD (3 Seeds)"
)
ax.set_ylim(0.96, 1.0)
ax.tick_params(axis="x", rotation=30)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(
    ANALYSIS_DIR / "configuration_comparison.png",
    dpi=300
)
plt.close(fig)


# ==========================================
# 8. SE VS CBAM PLACEMENT FIGURE
# ==========================================

fig, ax = plt.subplots(figsize=(7, 4))

for mechanism, label in [
    ("se", "SE"),
    ("cbam", "CBAM"),
]:
    selected = [
        next(
            r for r in summary
            if r["configuration"]
            == f"{mechanism}_{placement}"
        )
        for placement in PLACEMENTS
    ]

    ax.errorbar(
        ["Early", "Middle", "Late"],
        [r["mean_macro_f1"] for r in selected],
        yerr=[r["std_macro_f1"] for r in selected],
        marker="o",
        capsize=5,
        label=label
    )

ax.set_xlabel("Attention Placement")
ax.set_ylabel("Validation Macro-F1")
ax.set_title(
    "SE vs CBAM — Placement Comparison (Mean ± SD)"
)
ax.set_ylim(0.96, 1.0)
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(
    ANALYSIS_DIR / "placement_comparison.png",
    dpi=300
)
plt.close(fig)


# ==========================================
# 9. PRINT THE SUMMARY
# ==========================================

print("\n=== CONFIGURATION SUMMARY ===")

for row in summary:
    print(
        f"{row['configuration']:14s} "
        f"Macro-F1: "
        f"{row['mean_macro_f1']:.4f} "
        f"± {row['std_macro_f1']:.4f}"
    )

print("\n=== PLACEMENT RANKINGS BY SEED ===")

for row in rankings:
    print(
        f"{row['mechanism'].upper():4s} "
        f"seed {row['seed']}: "
        f"{row['rank_1']} > "
        f"{row['rank_2']} > "
        f"{row['rank_3']}"
    )

print("\nAll analysis files saved to:", ANALYSIS_DIR)
