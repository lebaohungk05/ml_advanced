import json
from pathlib import Path

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
QUERY_FILE = ROOT / "data" / "eval" / "queries.json"
sys.path.insert(0, str(ROOT))

from scripts.compute_baseline_results import load_labels, load_runs
from src.adapters.metrics.ranking_metrics import RankingMetrics

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
    }
)

COLORS = {
    "siglip2_lora": "#009e73",
    "siglip2": "#0072b2",
    "bm25": "#8c8c8c",
    "clip": "#e69f00",
    "visiglip_ot": "#cc79a7",
}
NAMES = {
    "siglip2_lora": "SigLIP 2 + DoRA",
    "siglip2": "SigLIP 2 gốc",
    "bm25": "BM25",
    "clip": "CLIP ViT-B/32",
    "visiglip_ot": "ViSigLIP-OT",
}
ORDER = ["bm25", "clip", "visiglip_ot", "siglip2", "siglip2_lora"]

GROUP_LABELS = ["A (46 câu)", "B (41 câu)", "C (33 câu)", "D (35 câu)"]


def comma(value, _position=None):
    return f"{value:.1f}".replace(".", ",")


def comma_formatter():
    return FuncFormatter(comma)


def decimal(value, digits=3):
    return f"{value:.{digits}f}".replace(".", ",")


def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def group_members():
    queries = json.loads(QUERY_FILE.read_text(encoding="utf-8"))["queries"]
    members = {}
    for query in queries:
        members.setdefault(query["group"], []).append(str(query["id"]))
    return [members[g] for g in "ABCD"]


def group_scores():
    labels = load_labels()
    runs = load_runs()
    metrics = RankingMetrics()
    recall, ndcg = {}, {}
    for system in ORDER:
        recall[system], ndcg[system] = [], []
        for members in group_members():
            sub = {q: runs[system][q] for q in members}
            recall[system].append(metrics.recall_at_k(sub, labels, 10))
            ndcg[system].append(metrics.ndcg_at_k(sub, labels, 10))
    return recall, ndcg


def fig_groups():
    group_recall, group_ndcg = group_scores()
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 3.5), sharex=True)
    width = 0.16
    x = np.arange(4)
    for ax, data, title in (
        (axes[0], group_recall, "Recall@10"),
        (axes[1], group_ndcg, "nDCG@10"),
    ):
        for index, system in enumerate(ORDER):
            ax.bar(x + (index - 2) * width, data[system], width, color=COLORS[system], label=NAMES[system], edgecolor="white", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(GROUP_LABELS, fontsize=8.5)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(title, fontsize=9)
        ax.yaxis.set_major_formatter(comma_formatter())
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="y", labelsize=8.5)
    axes[1].set_xlabel("Nhóm truy vấn")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=False, columnspacing=1.0, handlelength=1.2, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.8)
    save(fig, "fig_ch5_3")


if __name__ == "__main__":
    fig_groups()
