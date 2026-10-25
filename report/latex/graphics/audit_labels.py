import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / "data" / "eval"
OUT = Path(__file__).resolve().parent

SYSTEMS = ["bm25", "clip", "visiglip_ot", "siglip2", "siglip2_lora"]
ACCESSORY_HEADS = ["kính", "mũ", "băng đô", "cà vạt", "găng tay", "đồng hồ", "dây nịt", "thắt lưng", "vớ", "tất", "giày", "túi", "ví", "khăn", "dù"]
ACCESSORY_FILTER_NAMES = {"Kính mắt", "Mũ", "Băng đô", "Cà vạt", "Găng tay", "Đồng hồ", "Dây nịt", "Vớ", "Giày", "Túi", "Khăn choàng"}
ITEM_IN_TITLE = {"15": "Mũ", "18": "Găng tay", "22": "Vớ", "25": "Khăn choàng", "26": "Dù", "35": "Túi / ví", "73": "Mũ", "75": "Khăn choàng", "101": "Mũ", "40": "Khăn choàng"}


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def garment_names():
    categories = load_json(ROOT / "data" / "category_mapping.json")["categories"]
    return [
        c["vi"].split("/")[0].strip().split("(")[0].strip()
        for c in categories
        if c["group"] in ("main_garment", "accessory")
    ]


def keyword_in_query(text, names):
    text = text.lower()
    for name in names:
        for word in name.lower().split():
            if len(word) > 2 and word in text:
                return name
    return None


def load_labels():
    human = collections.defaultdict(dict)
    for path in glob.glob(str(EVAL / "grades_*.jsonl")):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                human[(row["query_id"], row["product_id"])][row["annotator"]] = row["grade"]
    auto = {}
    for name in ("auto_zero_labels", "auto_zero_labels_ft"):
        for row in load_json(EVAL / f"{name}.json"):
            auto[(row["query_id"], row["product_id"])] = row
    return human, auto


def grade_lists(human, auto):
    merged = {key: list(value.values()) for key, value in human.items()}
    for key in auto:
        merged.setdefault(key, [0])
    return merged


def evaluate(pool, grades, rule, queries, depth=10):
    totals = collections.Counter()
    per_query = {}
    for qid in queries:
        first = None
        for index, hit in enumerate(pool[qid][:depth], start=1):
            g = grades.get((qid, hit["product_id"]))
            if g and rule(g):
                first = index
                break
        per_query[qid] = first
        if first:
            totals["r1"] += first <= 1
            totals["r5"] += first <= 5
            totals["r10"] += first <= 10
            totals["mrr"] += 1.0 / first
    n = len(queries)
    return {k: totals[k] / n for k in ("r1", "r5", "r10", "mrr")}, per_query


def bootstrap(a, b, iterations=10000, seed=42):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = float(np.mean(a - b))
    if diff == 0.0:
        return diff, 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(iterations, len(a)))
    resampled = (a[idx] - b[idx]).mean(axis=1)
    extreme = np.count_nonzero(resampled <= 0) if diff > 0 else np.count_nonzero(resampled >= 0)
    return diff, min(2.0 * extreme / iterations, 1.0)


def main():
    names = garment_names()
    queries = {str(q["id"]): q for q in load_json(EVAL / "queries.json")["queries"]}
    catalog = {p["product_id"]: p for p in load_json(ROOT / "data" / "processed" / "test.json")}
    pool = load_json(EVAL / "pool_top20.json")
    human, auto = load_labels()
    grades = grade_lists(human, auto)
    result = {}

    rules = {
        "mean_ge_1": lambda g: sum(g) / len(g) >= 1,
        "any_ge_1": lambda g: max(g) >= 1,
        "all_ge_1": lambda g: min(g) >= 1,
        "mean_ge_1_5": lambda g: sum(g) / len(g) >= 1.5,
    }
    all_queries = list(queries)
    result["sensitivity"] = {}
    for rule_name, rule in rules.items():
        result["sensitivity"][rule_name] = {
            system: evaluate(pool[system], grades, rule, all_queries)[0] for system in SYSTEMS
        }

    accessory = {}
    for qid, q in queries.items():
        text = q["text"].lower()
        for head in sorted(ACCESSORY_HEADS, key=lambda x: -len(x)):
            if text.startswith(head):
                accessory[qid] = q["text"]
                break
    rows = []
    for qid, text in accessory.items():
        human_pairs = [k for k in human if k[0] == qid]
        auto_pairs = [k for k in auto if k[0] == qid and k not in human]
        top10 = [h["product_id"] for h in pool["siglip2_lora"][qid][:10]]
        rows.append(
            {
                "query_id": qid,
                "text": text,
                "human_pairs": len(human_pairs),
                "auto_pairs": len(auto_pairs),
                "lora_top10_auto": sum(1 for p in top10 if (qid, p) in auto and (qid, p) not in human),
                "lora_top10_human": sum(1 for p in top10 if (qid, p) in human),
                "lora_hit10": any(
                    (qid, p) in human and sum(human[(qid, p)].values()) / len(human[(qid, p)]) >= 1
                    for p in top10
                ),
            }
        )
    result["accessory_queries"] = rows

    inside = collections.Counter()
    outside = collections.Counter()
    for (qid, pid), g in human.items():
        if qid in ITEM_IN_TITLE:
            relevant = sum(g.values()) / len(g) >= 1
            (inside if ITEM_IN_TITLE[qid] in catalog[pid]["title"] else outside)[relevant] += 1
    result["human_acceptance_unfiltered_accessory"] = {
        "item_in_title": [inside[True], inside[True] + inside[False]],
        "item_not_in_title": [outside[True], outside[True] + outside[False]],
    }

    spurious = collections.Counter()
    clear_false = collections.Counter()
    containing = collections.Counter()
    for qid, pid in auto:
        q = queries[qid]["text"]
        keyword = keyword_in_query(q, names)
        first = keyword.lower().split()[0]
        if first not in q.lower():
            spurious[qid] += 1
        head = catalog[pid]["category"].split("/")[0].strip().split("(")[0].strip().lower()
        if head and head in q.lower():
            clear_false[qid] += 1
        if keyword in ACCESSORY_FILTER_NAMES and first in catalog[pid]["title"].lower():
            containing[qid] += 1
    result["auto_zero"] = {
        "total_rounds_1_2": len(auto),
        "spurious_keyword_pairs": sum(spurious.values()),
        "spurious_keyword_queries": len(spurious),
        "category_named_in_query_pairs": sum(clear_false.values()),
        "category_named_in_query_by_query": dict(clear_false),
        "accessory_filter_pairs_with_item_in_image": sum(containing.values()),
        "accessory_filter_queries": len(containing),
    }

    cases = []
    for case in load_json(EVAL / "error_cases.json")["cases"]:
        qid = str(case["query_id"])
        text = queries[qid]["text"]
        keyword = keyword_in_query(text, names)
        first = keyword.lower().split()[0]
        top10 = [h["product_id"] for h in pool["siglip2_lora"][qid][:10]]
        top5 = top10[:5]
        cases.append(
            {
                "query_id": qid,
                "text": text,
                "keyword": keyword,
                "keyword_in_query": first in text.lower(),
                "top5_auto": sum(1 for p in top5 if (qid, p) in auto and (qid, p) not in human),
                "top10_auto": sum(1 for p in top10 if (qid, p) in auto and (qid, p) not in human),
                "top1_source": "human" if (qid, top10[0]) in human else "auto",
            }
        )
    result["error_cases"] = cases

    excluded = {c["query_id"] for c in cases}
    kept = [q for q in all_queries if q not in excluded]
    rule = rules["mean_ge_1"]
    result["without_audited_queries"] = {"n_kept": len(kept), "n_excluded": len(excluded)}
    per_system = {}
    per_query_scores = {}
    for system in SYSTEMS:
        metrics, per_query = evaluate(pool[system], grades, rule, kept)
        per_system[system] = metrics
        per_query_scores[system] = per_query
    result["without_audited_queries"]["metrics"] = per_system
    tests = {}
    for name, a, b in (("siglip2_lora_vs_siglip2", "siglip2_lora", "siglip2"), ("siglip2_vs_bm25", "siglip2", "bm25"), ("bm25_vs_clip", "bm25", "clip")):
        r1_a = [1.0 if per_query_scores[a][q] == 1 else 0.0 for q in kept]
        r1_b = [1.0 if per_query_scores[b][q] == 1 else 0.0 for q in kept]
        r10_a = [1.0 if per_query_scores[a][q] else 0.0 for q in kept]
        r10_b = [1.0 if per_query_scores[b][q] else 0.0 for q in kept]
        rr_a = [1.0 / per_query_scores[a][q] if per_query_scores[a][q] else 0.0 for q in kept]
        rr_b = [1.0 / per_query_scores[b][q] if per_query_scores[b][q] else 0.0 for q in kept]
        tests[name] = {
            "recall@1": bootstrap(r1_a, r1_b),
            "recall@10": bootstrap(r10_a, r10_b),
            "mrr": bootstrap(rr_a, rr_b),
        }
    result["without_audited_queries"]["bootstrap"] = tests

    affected_rule = set(spurious) | set(containing)
    kept_rule = [q for q in all_queries if q not in affected_rule]
    per_system_rule = {}
    per_query_rule = {}
    for system in SYSTEMS:
        metrics, per_query = evaluate(pool[system], grades, rule, kept_rule)
        per_system_rule[system] = metrics
        per_query_rule[system] = per_query
    r1 = lambda s: [1.0 if per_query_rule[s][q] == 1 else 0.0 for q in kept_rule]
    r10 = lambda s: [1.0 if per_query_rule[s][q] else 0.0 for q in kept_rule]
    rr = lambda s: [1.0 / per_query_rule[s][q] if per_query_rule[s][q] else 0.0 for q in kept_rule]
    result["without_rule_affected_queries"] = {
        "n_kept": len(kept_rule),
        "n_excluded": len(affected_rule),
        "metrics": per_system_rule,
        "bootstrap_lora_vs_siglip2": {
            "recall@1": bootstrap(r1("siglip2_lora"), r1("siglip2")),
            "recall@10": bootstrap(r10("siglip2_lora"), r10("siglip2")),
            "mrr": bootstrap(rr("siglip2_lora"), rr("siglip2")),
        },
    }
    print(json.dumps(result["without_rule_affected_queries"], ensure_ascii=False, indent=1))

    with open(OUT / "audit_results.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=1)
    print(json.dumps(result["without_audited_queries"], ensure_ascii=False, indent=1))
    print(json.dumps(result["auto_zero"], ensure_ascii=False, indent=1))
    print(json.dumps(result["human_acceptance_unfiltered_accessory"]))


if __name__ == "__main__":
    main()
