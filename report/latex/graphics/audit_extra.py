import collections
import glob
import json
import sys

import audit_labels as audit

sys.stdout.reconfigure(encoding="utf-8")


def label_scale():
    rows = collections.Counter()
    by_annotator = collections.Counter()
    pair_annotators = collections.defaultdict(set)
    for path in glob.glob(str(audit.EVAL / "grades_*.jsonl")):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rows["rows"] += 1
                by_annotator[row["annotator"]] += 1
                pair_annotators[(row["query_id"], row["product_id"])].add(row["annotator"])
    distinct = collections.Counter(len(v) for v in pair_annotators.values())
    auto = {name: len(audit.load_json(audit.EVAL / f"{name}.json")) for name in ("auto_zero_labels", "auto_zero_labels_ft", "auto_zero_labels_axis2_siglip1")}
    return {
        "human_rows": rows["rows"],
        "human_rows_by_annotator": dict(by_annotator),
        "human_pairs": len(pair_annotators),
        "pairs_by_annotator_count": dict(distinct),
        "repeated_rows": rows["rows"] - sum(k * v for k, v in distinct.items()),
        "auto_pairs": auto,
    }


def acceptance_when_item_secondary():
    human, _ = audit.load_labels()
    catalog = {p["product_id"]: p for p in audit.load_json(audit.ROOT / "data" / "processed" / "test.json")}
    counter = collections.Counter()
    per_query = collections.defaultdict(lambda: [0, 0])
    for (qid, pid), grades in human.items():
        item = audit.ITEM_IN_TITLE.get(qid)
        if not item or item not in catalog[pid]["title"]:
            continue
        relevant = sum(grades.values()) / len(grades) >= 1
        main_is_item = item.split(" / ")[0].lower() in catalog[pid]["category"].lower()
        counter[(main_is_item, relevant)] += 1
        if not main_is_item:
            per_query[qid][1] += 1
            per_query[qid][0] += relevant
    return {
        "item_is_main_category": [counter[(True, True)], counter[(True, True)] + counter[(True, False)]],
        "item_is_secondary": [counter[(False, True)], counter[(False, True)] + counter[(False, False)]],
        "per_query_secondary": dict(per_query),
    }


def failures_inside_affected():
    queries = {str(q["id"]): q for q in audit.load_json(audit.EVAL / "queries.json")["queries"]}
    pool = audit.load_json(audit.EVAL / "pool_top20.json")
    human, auto = audit.load_labels()
    grades = audit.grade_lists(human, auto)
    names = audit.garment_names()
    catalog = {p["product_id"]: p for p in audit.load_json(audit.ROOT / "data" / "processed" / "test.json")}
    spurious, containing = set(), set()
    for qid, pid in auto:
        text = queries[qid]["text"]
        keyword = audit.keyword_in_query(text, names)
        first = keyword.lower().split()[0]
        if first not in text.lower():
            spurious.add(qid)
        if keyword in audit.ACCESSORY_FILTER_NAMES and first in catalog[pid]["title"].lower():
            containing.add(qid)
    affected = spurious | containing
    rule = lambda g: sum(g) / len(g) >= 1
    result = {"affected": len(affected), "spurious": len(spurious), "accessory_filter": len(containing)}
    for system in ("siglip2", "siglip2_lora"):
        _, per_query = audit.evaluate(pool[system], grades, rule, list(queries))
        failed = [q for q in queries if not per_query[q]]
        result[system] = {"failed": len(failed), "failed_inside_affected": len([q for q in failed if q in affected])}
    return result


if __name__ == "__main__":
    output = {
        "label_scale": label_scale(),
        "acceptance_when_item_secondary": acceptance_when_item_secondary(),
        "failures_inside_affected": failures_inside_affected(),
    }
    with open(audit.OUT / "audit_extra.json", "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=1)
    print(json.dumps(output, ensure_ascii=False, indent=1))
