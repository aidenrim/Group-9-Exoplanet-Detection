"""
Plot a grouped bar chart comparing models across precision, accuracy, F1,
and AUC-ROC.

Reads results/{model}/eval.json (written by scripts/evaluate.py) for each
specified model and saves the chart to results/comparison.png.

Usage:
    python scripts/compare_models.py --models run_v1 run_v2 run_v3
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

METRICS       = ["precision", "accuracy", "f1", "auc"]
METRIC_LABELS = ["Precision", "Accuracy", "F1", "AUC-ROC"]

# One colour per model — extend if more than 8 models are compared.
MODEL_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grouped bar chart comparing models across key metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        metavar="NAME",
        help="One or more model names to compare.",
    )
    args = parser.parse_args()

    if len(args.models) > len(MODEL_COLORS):
        log.error(f"At most {len(MODEL_COLORS)} models are currently supported.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Load eval.json for each model
    # ------------------------------------------------------------------
    records: list[dict] = []
    for model_name in args.models:
        json_path = ROOT / "results" / model_name / "eval.json"
        if not json_path.exists():
            log.error(
                f"Eval file not found: {json_path}\n"
                f"Run: python scripts/evaluate.py --model {model_name} --dataset <dataset>"
            )
            sys.exit(1)
        with json_path.open() as f:
            records.append(json.load(f))
        log.info(f"  Loaded {json_path}")

    # ------------------------------------------------------------------
    # 2. Build the grouped bar chart
    # ------------------------------------------------------------------
    n_models  = len(records)
    n_metrics = len(METRICS)
    bar_width = 0.8 / n_models
    x         = np.arange(n_metrics)   # one group per metric

    fig, ax = plt.subplots(figsize=(max(7, 2 * n_metrics + n_models), 5))

    for i, (record, model_name) in enumerate(zip(records, args.models)):
        values  = [record[metric] for metric in METRICS]
        offsets = (i - (n_models - 1) / 2) * bar_width
        bars    = ax.bar(
            x + offsets,
            values,
            width=bar_width,
            label=model_name,
            color=MODEL_COLORS[i],
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.09)
    ax.set_title("Model comparison")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    out_path = ROOT / "results" / "comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info(f"Chart saved to {out_path}")


if __name__ == "__main__":
    main()
