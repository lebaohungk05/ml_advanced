import json
import collections
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "data"

PALETTE = {
    "bm25": "#8c8c8c",
    "clip": "#e69f00",
    "siglip2": "#0072b2",
    "visiglip_ot": "#cc79a7",
    "siglip2_lora": "#009e73",
    "axis1": "#56b4e9",
    "axis2": "#d55e00",
}
LABELS = {
    "bm25": "BM25",
    "clip": "CLIP ViT-B/32",
    "siglip2": "SigLIP 2 (zero-shot)",
    "visiglip_ot": "ViSigLIP-OT",
    "siglip2_lora": "SigLIP 2 + DoRA",
}
SYSTEMS = ["bm25", "clip", "visiglip_ot", "siglip2", "siglip2_lora"]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def annotate_bars(ax, bars, fmt="{:.2f}", size=7.5, dy=0.01):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + dy,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=size,
        )


def grouped_bars(ax, series, categories, colors, width=0.16, fmt="{:.2f}", size=6.5):
    count = len(series)
    positions = np.arange(len(categories))
    for index, (label, values) in enumerate(series.items()):
        offset = (index - (count - 1) / 2) * width
        bars = ax.bar(
            positions + offset,
            values,
            width,
            label=label,
            color=colors[index],
            edgecolor="white",
            linewidth=0.5,
        )
        annotate_bars(ax, bars, fmt=fmt, size=size)
    ax.set_xticks(positions)
    ax.set_xticklabels(categories)


def load_catalog():
    with open(DATA / "processed" / "test.json", encoding="utf-8") as handle:
        return json.load(handle)


def fig_catalog_categories():
    catalog = load_catalog()
    counts = collections.Counter(p["category"] for p in catalog).most_common()
    names = [name for name, _ in counts][::-1]
    values = [value for _, value in counts][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.barh(names, values, color="#0072b2")
    ax.set_xscale("log")
    ax.set_xlabel("Số ảnh trong tập kiểm tra (thang log)")
    for bar, value in zip(bars, values):
        ax.text(value * 1.08, bar.get_y() + bar.get_height() / 2, f"{value:,}".replace(",", "."), va="center", fontsize=8)
    ax.set_xlim(0.8, max(values) * 3)
    ax.grid(axis="y", visible=False)
    save(fig, "fig_catalog_categories")


def fig_items_per_image():
    catalog = load_catalog()
    counter = collections.Counter(len([t for t in p["title"].split(", ") if t]) for p in catalog)
    keys = sorted(counter)
    values = [counter[k] for k in keys]
    total = sum(values)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bars = ax.bar([str(k) for k in keys], values, color="#d55e00")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 20, f"{100 * value / total:.1f}%", ha="center", fontsize=8)
    ax.set_xlabel("Số món trang phục hoặc phụ kiện được gắn nhãn trong một ảnh")
    ax.set_ylabel("Số ảnh")
    ax.grid(axis="x", visible=False)
    save(fig, "fig_items_per_image")


def fig_query_groups():
    with open(DATA / "eval" / "queries.json", encoding="utf-8") as handle:
        queries = json.load(handle)["queries"]
    groups = ["A", "B", "C", "D"]
    counts = [sum(1 for q in queries if q["group"] == g) for g in groups]
    lengths = [np.mean([len(q["text"].split()) for q in queries if q["group"] == g]) for g in groups]
    names = ["A: đơn thuộc tính", "B: đa thuộc tính", "C: từ vựng mượn", "D: phong cách"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    bars = axes[0].bar(names, counts, color=["#0072b2", "#009e73", "#e69f00", "#cc79a7"])
    annotate_bars(axes[0], bars, fmt="{:.0f}", size=9, dy=0.5)
    axes[0].set_ylabel("Số truy vấn")
    axes[0].set_ylim(0, 55)
    axes[0].tick_params(axis="x", labelrotation=20)
    axes[0].grid(axis="x", visible=False)
    bars = axes[1].bar(names, lengths, color=["#0072b2", "#009e73", "#e69f00", "#cc79a7"])
    annotate_bars(axes[1], bars, fmt="{:.1f}", size=9, dy=0.1)
    axes[1].set_ylabel("Số âm tiết trung bình mỗi truy vấn")
    axes[1].set_ylim(0, 10.5)
    axes[1].tick_params(axis="x", labelrotation=20)
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig_query_groups")


def fig_grades():
    counts = {
        "Hiệp": (2564, 548, 1358),
        "Hùng": (2169, 606, 1254),
        "Hưng": (1136, 302, 264),
    }
    names = list(counts)
    totals = [sum(counts[n]) for n in names]
    shares = np.array([[c / t for c in counts[n]] for n, t in zip(names, totals)])
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    left = np.zeros(len(names))
    colors = ["#999999", "#e69f00", "#009e73"]
    labels = ["Điểm 0", "Điểm 1", "Điểm 2"]
    for index in range(3):
        bars = ax.barh(names, shares[:, index], left=left, color=colors[index], label=labels[index])
        for bar, share in zip(bars, shares[:, index]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, f"{100 * share:.0f}%", ha="center", va="center", fontsize=8, color="white")
        left += shares[:, index]
    for position, total in enumerate(totals):
        ax.text(1.01, position, f"n = {total:,}".replace(",", "."), va="center", fontsize=8)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Tỉ lệ các mức điểm do mỗi người chấm")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.45, 1.25), frameon=False)
    ax.grid(axis="y", visible=False)
    save(fig, "fig_grades")


def fig_main_results():
    metrics = {
        "bm25": (0.2000, 0.4710, 0.5677),
        "clip": (0.1677, 0.3806, 0.4258),
        "visiglip_ot": (0.0839, 0.2258, 0.2774),
        "siglip2": (0.4903, 0.7742, 0.8645),
        "siglip2_lora": (0.6968, 0.8516, 0.8839),
    }
    series = {LABELS[s]: [metrics[s][i] for i in range(3)] for s in SYSTEMS}
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    grouped_bars(ax, series, ["Recall@1", "Recall@5", "Recall@10"], [PALETTE[s] for s in SYSTEMS], width=0.15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Tỉ lệ truy vấn có kết quả đúng")
    ax.legend(ncol=3, loc="upper left", frameon=False, fontsize=8)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_main_recall")

    rank_metrics = {
        "bm25": (0.3175, 0.1785),
        "clip": (0.2516, 0.1394),
        "visiglip_ot": (0.1450, 0.0662),
        "siglip2": (0.6062, 0.4486),
        "siglip2_lora": (0.7659, 0.6843),
    }
    series = {LABELS[s]: [rank_metrics[s][i] for i in range(2)] for s in SYSTEMS}
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    grouped_bars(ax, series, ["MRR", "nDCG@10"], [PALETTE[s] for s in SYSTEMS], width=0.15, size=7)
    ax.set_ylim(0, 0.95)
    ax.legend(ncol=2, loc="upper left", frameon=False, fontsize=8)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_main_rank")


def fig_groups():
    ndcg = {
        "bm25": (0.2343, 0.1291, 0.0931, 0.2421),
        "clip": (0.0274, 0.0712, 0.1600, 0.3134),
        "visiglip_ot": (0.0121, 0.0258, 0.0115, 0.2134),
        "siglip2": (0.4591, 0.3747, 0.3421, 0.6043),
        "siglip2_lora": (0.6351, 0.6246, 0.6648, 0.8159),
    }
    recall = {
        "bm25": (0.6522, 0.4146, 0.2121, 0.9714),
        "clip": (0.1304, 0.2927, 0.5455, 0.8571),
        "visiglip_ot": (0.0870, 0.1463, 0.0909, 0.8571),
        "siglip2": (0.8043, 0.8293, 0.8485, 1.0000),
        "siglip2_lora": (0.8043, 0.8537, 0.9091, 1.0000),
    }
    groups = ["A (n=46)", "B (n=41)", "C (n=33)", "D (n=35)"]
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.6), sharex=True)
    for ax, data, title in zip(axes, (recall, ndcg), ("Recall@10", "nDCG@10")):
        series = {LABELS[s]: data[s] for s in SYSTEMS}
        grouped_bars(ax, series, groups, [PALETTE[s] for s in SYSTEMS], width=0.15, size=6)
        ax.set_ylabel(title)
        ax.set_ylim(0, 1.12)
        ax.grid(axis="x", visible=False)
    axes[0].legend(ncol=3, loc="upper left", frameon=False, fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig_groups")


def fig_accessory():
    recall = {
        "bm25": (0.3478, 0.6061),
        "clip": (0.0435, 0.4924),
        "visiglip_ot": (0.0435, 0.3182),
        "siglip2": (0.3913, 0.9470),
        "siglip2_lora": (0.3913, 0.9697),
    }
    ndcg = {
        "bm25": (0.3101, 0.1683),
        "clip": (0.0444, 0.1467),
        "visiglip_ot": (0.0049, 0.0710),
        "siglip2": (0.4303, 0.4500),
        "siglip2_lora": (0.4526, 0.7022),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    for ax, data, title in zip(axes, (recall, ndcg), ("Recall@10", "nDCG@10")):
        series = {LABELS[s]: data[s] for s in SYSTEMS}
        grouped_bars(ax, series, ["Phụ kiện\n(23 truy vấn)", "Trang phục chính\n(132 truy vấn)"], [PALETTE[s] for s in SYSTEMS], width=0.15, size=6)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(axis="x", visible=False)
    axes[0].legend(ncol=2, loc="upper left", frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig_accessory")

    delta_labels = ["Nhóm A", "Nhóm B", "Nhóm C", "Nhóm D", "Phụ kiện", "Trang phục\nchính"]
    delta_ndcg = [0.1760, 0.2499, 0.3227, 0.2116, 0.0223, 0.2522]
    delta_recall = [0.0, 0.0244, 0.0606, 0.0, 0.0, 0.0227]
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    positions = np.arange(len(delta_labels))
    bars1 = ax.bar(positions - 0.18, delta_recall, 0.36, label="Δ Recall@10", color="#56b4e9")
    bars2 = ax.bar(positions + 0.18, delta_ndcg, 0.36, label="Δ nDCG@10", color="#009e73")
    annotate_bars(ax, bars1, fmt="{:+.3f}", size=7.5, dy=0.006)
    annotate_bars(ax, bars2, fmt="{:+.3f}", size=7.5, dy=0.006)
    ax.set_xticks(positions)
    ax.set_xticklabels(delta_labels)
    ax.set_ylabel("Mức tăng của SigLIP 2 + DoRA\nso với SigLIP 2 zero-shot")
    ax.set_ylim(0, 0.38)
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_delta")


def fig_latency():
    systems = ["bm25", "clip", "siglip2", "visiglip_ot", "siglip2_lora"]
    encode = {"bm25": 0.01, "clip": 40.83, "siglip2": 48.94, "visiglip_ot": 71.38, "siglip2_lora": 193.59}
    search = {"bm25": 8.68, "clip": 26.37, "siglip2": 41.24, "visiglip_ot": 48.12, "siglip2_lora": 48.61}
    total_p50 = {"bm25": 14.67, "clip": 66.78, "siglip2": 91.55, "visiglip_ot": 118.25, "siglip2_lora": 245.57}
    total_p95 = {"bm25": 24.13, "clip": 111.96, "siglip2": 127.37, "visiglip_ot": 146.45, "siglip2_lora": 746.32}
    names = [LABELS[s] for s in systems]
    positions = np.arange(len(systems))
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    axes[0].bar(positions, [encode[s] for s in systems], 0.55, label="Mã hoá truy vấn", color="#0072b2")
    axes[0].bar(positions, [search[s] for s in systems], 0.55, bottom=[encode[s] for s in systems], label="Tìm trong chỉ mục", color="#e69f00")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    axes[0].set_ylabel("Độ trễ p50 (ms)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].grid(axis="x", visible=False)
    width = 0.36
    b1 = axes[1].bar(positions - width / 2, [total_p50[s] for s in systems], width, label="p50", color="#009e73")
    b2 = axes[1].bar(positions + width / 2, [total_p95[s] for s in systems], width, label="p95", color="#d55e00")
    annotate_bars(axes[1], b1, fmt="{:.0f}", size=7, dy=8)
    annotate_bars(axes[1], b2, fmt="{:.0f}", size=7, dy=8)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    axes[1].set_ylabel("Độ trễ đầu-cuối (ms)")
    axes[1].set_ylim(0, 840)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig_latency")


def fig_ablation():
    systems = ["SigLIP 2\nzero-shot", "DoRA +\nhard-neg\n(tham chiếu)", "LoRA thường\n(axis 1)", "DoRA, không\nhard-neg\n(axis 2)"]
    recall1 = [0.4903, 0.6968, 0.6903, 0.6968]
    recall10 = [0.8645, 0.8839, 0.8839, 0.8581]
    mrr = [0.6062, 0.7659, 0.7599, 0.7629]
    colors = ["#0072b2", "#009e73", "#56b4e9", "#d55e00"]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6), sharey=False)
    for ax, values, title, limits in zip(axes, (recall1, recall10, mrr), ("Recall@1", "Recall@10", "MRR"), ((0, 0.9), (0.5, 1.0), (0.3, 0.9))):
        bars = ax.bar(range(4), values, color=colors)
        annotate_bars(ax, bars, fmt="{:.4f}", size=7, dy=0.005)
        ax.set_xticks(range(4))
        ax.set_xticklabels(systems, fontsize=6.5)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(*limits)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig_ablation")


def fig_siglip1_interval():
    names = ["SigLIP 1\n(khoảng)", "SigLIP 2\nzero-shot", "SigLIP 2\n+ DoRA"]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    metrics = [
        ("Recall@1", (0.3161, 0.807), 0.4903, 0.6968),
        ("Recall@5", (0.5484, 0.987), 0.7742, 0.8516),
        ("Recall@10", (0.7097, 1.000), 0.8645, 0.8839),
    ]
    for ax, (title, interval, zero_shot, tuned) in zip(axes, metrics):
        ax.bar([0], [interval[1] - interval[0]], bottom=[interval[0]], width=0.5, color="#cc79a7", alpha=0.85)
        ax.bar([1], [zero_shot], width=0.5, color="#0072b2")
        ax.bar([2], [tuned], width=0.5, color="#009e73")
        ax.text(0, interval[1] + 0.012, f"[{interval[0]:.3f};\n{interval[1]:.3f}]", ha="center", fontsize=7)
        ax.text(1, zero_shot + 0.012, f"{zero_shot:.3f}", ha="center", fontsize=8)
        ax.text(2, tuned + 0.012, f"{tuned:.3f}", ha="center", fontsize=8)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(names, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.18)
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig_siglip1_interval")


def fig_errors():
    labels = [
        "Ảnh chứa nhiều sản phẩm",
        "Phong cách / hoàn cảnh trừu tượng",
        "Sai thuộc tính màu sắc",
        "Sai chất liệu",
        "Watermark / chữ / collage",
        "Từ vựng bản địa / mượn",
    ]
    values = [14, 3, 1, 1, 1, 0]
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    positions = np.arange(len(labels))[::-1]
    bars = ax.barh(positions, values, color=["#d55e00", "#cc79a7", "#e69f00", "#e69f00", "#999999", "#0072b2"])
    for bar, value in zip(bars, values):
        ax.text(value + 0.2, bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=9)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Số ca trong 20 ca lỗi tệ nhất")
    ax.set_xlim(0, 16)
    ax.grid(axis="y", visible=False)
    save(fig, "fig_errors")


def fig_kappa():
    names = ["Hiệp – Hùng\n(3.545 dòng)", "Hiệp – Hưng\n(1.219 dòng)", "Hùng – Hưng\n(779 dòng)"]
    values = [0.520, 0.442, 0.441]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    bars = ax.bar(names, values, color="#0072b2", width=0.5)
    annotate_bars(ax, bars, fmt="{:.3f}", size=9, dy=0.008)
    ax.axhspan(0.41, 0.60, color="#e69f00", alpha=0.12)
    ax.text(2.45, 0.595, "Mức vừa phải\n(0,41–0,60)", ha="right", va="top", fontsize=7.5)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("Hệ số Cohen's κ")
    ax.grid(axis="x", visible=False)
    save(fig, "fig_kappa")


def fig_training():
    runs = ["hung-run1\n(DoRA, hard-neg)", "axis1\n(LoRA thường)", "axis2\n(không hard-neg)"]
    proxy = [0.2652, 0.2614, 0.2828]
    hours = [11084.7 / 3600, 7437.6 / 3600, 8642.1 / 3600]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    bars = axes[0].bar(runs, proxy, color=["#009e73", "#56b4e9", "#d55e00"], width=0.55)
    annotate_bars(axes[0], bars, fmt="{:.4f}", size=8, dy=0.003)
    axes[0].set_ylim(0, 0.34)
    axes[0].set_ylabel("Recall@5 self-query trên tập val")
    axes[0].tick_params(axis="x", labelsize=8)
    axes[0].grid(axis="x", visible=False)
    bars = axes[1].bar(runs, hours, color=["#009e73", "#56b4e9", "#d55e00"], width=0.55)
    annotate_bars(axes[1], bars, fmt="{:.2f} giờ", size=8, dy=0.05)
    axes[1].set_ylim(0, 3.7)
    axes[1].set_ylabel("Thời gian huấn luyện (giờ)")
    axes[1].tick_params(axis="x", labelsize=8)
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig_training")


def fig_pool_sizes():
    stages = ["Vòng 1\n4 baseline\ntop-10", "Vòng 2\nSigLIP 2+DoRA\ntop-10", "Vòng 3 (chưa chấm)\naxis 2 + SigLIP 1\ntop-10"]
    human = [4011, 940, 859]
    auto = [2068, 305, 403]
    positions = np.arange(3)
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    b1 = ax.bar(positions, human, 0.5, label="Cặp cần người chấm", color="#0072b2")
    b2 = ax.bar(positions, auto, 0.5, bottom=human, label="Cặp tự động chấm 0 (lọc danh mục)", color="#999999")
    for bar, value in zip(b1, human):
        ax.text(bar.get_x() + bar.get_width() / 2, value / 2, f"{value:,}".replace(",", "."), ha="center", va="center", color="white", fontsize=9)
    for bar, base, value in zip(b2, human, auto):
        ax.text(bar.get_x() + bar.get_width() / 2, base + value / 2, f"{value:,}".replace(",", "."), ha="center", va="center", color="white", fontsize=9)
    ax.set_xticks(positions)
    ax.set_xticklabels(stages, fontsize=8.5)
    ax.set_ylabel("Số cặp (truy vấn, ảnh)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_pool_sizes")


def fig_audit():
    with open(OUT / "audit_results.json", encoding="utf-8") as handle:
        audit = json.load(handle)
    cases = audit["error_cases"]
    labels = [f"{c['query_id']}. {c['text'][:34]}" for c in cases][::-1]
    auto = [c["top10_auto"] for c in cases][::-1]
    human = [10 - c["top10_auto"] for c in cases][::-1]
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    positions = np.arange(len(cases))
    ax.barh(positions, auto, color="#999999", label="Nhãn 0 do bộ lọc tự động")
    ax.barh(positions, human, left=auto, color="#0072b2", label="Nhãn do người chấm")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlim(0, 10)
    ax.set_xlabel("Số kết quả trong top-10 của SigLIP 2 + DoRA")
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.grid(axis="y", visible=False)
    save(fig, "fig_audit_cases")

    accept = audit["human_acceptance_unfiltered_accessory"]
    inside = accept["item_in_title"]
    outside = accept["item_not_in_title"]
    values = [100 * inside[0] / inside[1], 100 * outside[0] / outside[1]]
    names = [
        f"Ảnh có món được hỏi\n({inside[0]}/{inside[1]})",
        f"Ảnh không có món đó\n({outside[0]}/{outside[1]})",
    ]
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    bars = ax.bar(names, values, color=["#009e73", "#999999"], width=0.5)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.1f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 75)
    ax.set_ylabel("Tỉ lệ cặp được người chấm\ncho là liên quan")
    ax.grid(axis="x", visible=False)
    save(fig, "fig_human_acceptance")


if __name__ == "__main__":
    fig_catalog_categories()
    fig_items_per_image()
    fig_query_groups()
    fig_grades()
    fig_main_results()
    fig_groups()
    fig_accessory()
    fig_latency()
    fig_ablation()
    fig_siglip1_interval()
    fig_errors()
    fig_kappa()
    fig_training()
    fig_pool_sizes()
    fig_audit()
