import collections
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import audit_labels as audit
from make_figures import PALETTE, LABELS, OUT, save

sys.stdout.reconfigure(encoding="utf-8")

ORDER = ["siglip2_lora", "siglip2", "bm25", "clip", "visiglip_ot"]
PAIRS = [
    ("siglip2_lora", "siglip2"),
    ("siglip2", "bm25"),
    ("siglip2", "clip"),
    ("bm25", "clip"),
    ("bm25", "visiglip_ot"),
    ("clip", "visiglip_ot"),
]


def affected_queries(queries, auto, catalog, names):
    spurious, containing = set(), set()
    for qid, pid in auto:
        text = queries[qid]["text"]
        keyword = audit.keyword_in_query(text, names)
        first = keyword.lower().split()[0]
        if first not in text.lower():
            spurious.add(qid)
        if keyword in audit.ACCESSORY_FILTER_NAMES and first in catalog[pid]["title"].lower():
            containing.add(qid)
    return spurious | containing


def collect():
    queries = {str(q["id"]): q for q in audit.load_json(audit.EVAL / "queries.json")["queries"]}
    catalog = {p["product_id"]: p for p in audit.load_json(audit.ROOT / "data" / "processed" / "test.json")}
    pool = audit.load_json(audit.EVAL / "pool_top20.json")
    human, auto = audit.load_labels()
    grades = audit.grade_lists(human, auto)
    affected = affected_queries(queries, auto, catalog, audit.garment_names())
    kept = [q for q in queries if q not in affected]
    rule = lambda g: sum(g) / len(g) >= 1
    result = {"n_all": len(queries), "n_kept": len(kept), "n_affected": len(affected)}
    first_rank = {}
    for system in ORDER:
        full, per_all = audit.evaluate(pool[system], grades, rule, list(queries))
        part, per_kept = audit.evaluate(pool[system], grades, rule, kept)
        first_rank[system] = per_all
        result.setdefault("metrics", {})[system] = {"all": full, "kept": part}

    def vectors(system, kind):
        ranks = [first_rank[system][q] for q in kept]
        if kind == "r1":
            return [1.0 if r == 1 else 0.0 for r in ranks]
        if kind == "r10":
            return [1.0 if r else 0.0 for r in ranks]
        return [1.0 / r if r else 0.0 for r in ranks]

    result["bootstrap"] = {}
    for a, b in PAIRS:
        result["bootstrap"][f"{a}|{b}"] = {
            kind: audit.bootstrap(vectors(a, kind), vectors(b, kind)) for kind in ("r1", "r10", "mrr")
        }

    groups = collections.defaultdict(list)
    for q in kept:
        groups[queries[q]["group"]].append(q)
    result["groups"] = {}
    for group, members in sorted(groups.items()):
        result["groups"][group] = {"n": len(members)}
        for system in ORDER:
            ranks = [first_rank[system][q] for q in members]
            result["groups"][group][system] = {
                "hits": sum(1 for r in ranks if r),
                "recall10": sum(1 for r in ranks if r) / len(members),
                "recall1": sum(1 for r in ranks if r == 1) / len(members),
            }
    return result


def fig_before_after(result):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True)
    labels = [LABELS[s] for s in ORDER]
    y = np.arange(len(ORDER))[::-1]
    for ax, key, title in ((axes[0], "r1", "Recall@1"), (axes[1], "r10", "Recall@10")):
        before = [result["metrics"][s]["all"][key] for s in ORDER]
        after = [result["metrics"][s]["kept"][key] for s in ORDER]
        for yi, b, a, s in zip(y, before, after, ORDER):
            ax.plot([b, a], [yi, yi], color=PALETTE[s], linewidth=2, alpha=0.6)
            ax.scatter([b], [yi], color="white", edgecolor=PALETTE[s], s=44, zorder=3)
            ax.scatter([a], [yi], color=PALETTE[s], s=44, zorder=3)
            ax.text(a + 0.025, yi, f"{a:.3f}".replace(".", ","), va="center", fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, 1.15)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.grid(axis="y", visible=False)
    axes[0].scatter([], [], color="white", edgecolor="#555555", s=44, label="155 truy vấn")
    axes[0].scatter([], [], color="#555555", s=44, label="120 truy vấn còn lại")
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_ch5_before_after")


if __name__ == "__main__":
    data = collect()
    with open(OUT / "ch5_reconciled.json", "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=1)
    fig_before_after(data)
    print(json.dumps(data, ensure_ascii=False, indent=1))
