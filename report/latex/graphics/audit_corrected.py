import json
import sys

import audit_labels as audit

sys.stdout.reconfigure(encoding="utf-8")

DOUBLE = "Váy đầm"
SKIRT = "Chân váy"
INTENDED = {
    "11": [DOUBLE], "14": ["Kính mắt"], "16": ["Băng đô"], "17": ["Cà vạt"], "18": ["Găng tay"],
    "19": ["Đồng hồ"], "20": ["Dây nịt"], "23": ["Giày"], "24": ["Túi"], "25": ["Khăn choàng"],
    "29": [DOUBLE], "35": ["Túi"], "36": ["Giày"], "40": ["Khăn choàng"], "41": ["Áo sơ mi"],
    "44": [DOUBLE], "50": [SKIRT], "54": [DOUBLE], "57": [DOUBLE], "64": [DOUBLE],
    "69": [DOUBLE], "71": ["Giày"], "72": ["Túi"], "73": ["Mũ"], "74": ["Kính mắt"],
    "75": ["Khăn choàng"], "79": [DOUBLE], "83": [DOUBLE, SKIRT], "91": [DOUBLE, SKIRT],
    "95": [DOUBLE], "99": ["Giày"], "100": ["Túi"], "139": [DOUBLE], "143": [DOUBLE],
    "148": [DOUBLE],
}
SYSTEMS = ["siglip2_lora", "siglip2", "bm25", "clip", "visiglip_ot"]


def title_items(title):
    return [part.split("/")[0].strip() for part in title.split(",")]


def mean_relevant(grade_list):
    return sum(grade_list) / len(grade_list) >= 1


def upper_grades(grades, auto, human, catalog):
    upgraded = dict(grades)
    count = 0
    for (qid, pid) in auto:
        if (qid, pid) in human or qid not in INTENDED:
            continue
        items = title_items(catalog[pid]["title"])
        if any(name in items for name in INTENDED[qid]):
            upgraded[(qid, pid)] = [1]
            count += 1
    return upgraded, count


def first_hits(pool, grades, queries, depth=10):
    result = {}
    for qid in queries:
        result[qid] = None
        for index, hit in enumerate(pool[qid][:depth], start=1):
            g = grades.get((qid, hit["product_id"]))
            if g and mean_relevant(g):
                result[qid] = index
                break
    return result


def rates(hits, queries):
    n = len(queries)
    return {
        "r1": sum(1 for q in queries if hits[q] == 1) / n,
        "r5": sum(1 for q in queries if hits[q] and hits[q] <= 5) / n,
        "r10": sum(1 for q in queries if hits[q]) / n,
        "failed": sum(1 for q in queries if not hits[q]),
    }


def main():
    catalog = {p["product_id"]: p for p in audit.load_json(audit.ROOT / "data" / "processed" / "test.json")}
    pool = audit.load_json(audit.EVAL / "pool_top20.json")
    queries = [str(q["id"]) for q in audit.load_json(audit.EVAL / "queries.json")["queries"]]
    human, auto = audit.load_labels()
    grades = audit.grade_lists(human, auto)
    upper, upgraded = upper_grades(grades, auto, human, catalog)
    affected = [q for q in queries if q in INTENDED]
    out = {"n_affected": len(affected), "pairs_upgraded": upgraded, "systems": {}}
    per_query = {}
    for system in SYSTEMS:
        low = first_hits(pool[system], grades, queries)
        high = first_hits(pool[system], upper, queries)
        per_query[system] = (low, high)
        out["systems"][system] = {
            "all_low": rates(low, queries),
            "all_high": rates(high, queries),
            "affected_low": rates(low, affected),
            "affected_high": rates(high, affected),
        }
    lora_low, lora_high = per_query["siglip2_lora"]
    failed = [q for q in queries if not lora_low[q]]
    out["lora_failed"] = len(failed)
    out["lora_failed_recovered"] = len([q for q in failed if lora_high[q]])
    out["lora_failed_still"] = [q for q in failed if not lora_high[q]]
    zs_low, zs_high = per_query["siglip2"]
    zfailed = [q for q in queries if not zs_low[q]]
    out["zs_failed"] = len(zfailed)
    out["zs_failed_recovered"] = len([q for q in zfailed if zs_high[q]])
    a = [1.0 if lora_high[q] == 1 else 0.0 for q in queries]
    b = [1.0 if zs_high[q] == 1 else 0.0 for q in queries]
    out["bootstrap_high_r1_lora_vs_zs"] = audit.bootstrap(a, b)
    a = [1.0 if lora_high[q] else 0.0 for q in queries]
    b = [1.0 if zs_high[q] else 0.0 for q in queries]
    out["bootstrap_high_r10_lora_vs_zs"] = audit.bootstrap(a, b)
    order = {m: [s for s in sorted(SYSTEMS, key=lambda s: -out["systems"][s]["all_high"][m])] for m in ("r1", "r5", "r10")}
    out["order_high"] = order
    with open(audit.OUT / "audit_corrected.json", "w", encoding="utf-8") as handle:
        json.dump(out, handle, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
