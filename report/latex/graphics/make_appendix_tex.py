import collections
import sys
from pathlib import Path

import audit_labels as audit

sys.stdout.reconfigure(encoding="utf-8")

CHAPTERS = Path(__file__).resolve().parents[1] / "chapters"
QUERIES_OUTPUT = CHAPTERS / "phu_luc_truy_van.tex"
AUDIT_OUTPUT = CHAPTERS / "phu_luc_doi_soat.tex"

ESCAPES = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "$": r"\$", "{": r"\{", "}": r"\}"}

FILTER_KIND = {"spurious": "Từ khoá giả", "dress": "Từ khoá đầm", "accessory": "Bộ lọc phụ kiện"}
DRESS_KEYWORD = "Váy đầm"


def escape(text):
    return "".join(ESCAPES.get(ch, ch) for ch in text)


def queries_table(queries):
    head = r"\textbf{Mã} & \textbf{Nhóm} & \textbf{Truy vấn} & \textbf{Mã} & \textbf{Nhóm} & \textbf{Truy vấn} \\"
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{0.9}",
        r"\begin{longtable}{@{}r c p{5.7cm} r c p{5.7cm}@{}}",
        r"\caption{Bộ 155 truy vấn đánh giá}\label{tab:pl-truy-van}\\",
        r"\toprule",
        head,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        head,
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    half = (len(queries) + 1) // 2
    for index in range(half):
        left = queries[index]
        cells = [str(left["id"]), left["group"], escape(left["text"])]
        if index + half < len(queries):
            right = queries[index + half]
            cells += [str(right["id"]), right["group"], escape(right["text"])]
        else:
            cells += ["", "", ""]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\end{longtable}", r"\endgroup"]
    return "\n".join(lines)


def filter_statistics(queries, catalog, names, human, auto):
    spurious = collections.Counter()
    containing = collections.Counter()
    keyword_of = {}
    for qid, pid in auto:
        text = queries[qid]["text"]
        keyword = audit.keyword_in_query(text, names)
        first = keyword.lower().split()[0]
        keyword_of[qid] = keyword
        if first not in text.lower():
            spurious[qid] += 1
        if keyword in audit.ACCESSORY_FILTER_NAMES and first in catalog[pid]["title"].lower():
            containing[qid] += 1
    return spurious, containing, keyword_of


def lora_top10_auto(pool, human, auto):
    counts = {}
    for qid, hits in pool["siglip2_lora"].items():
        top10 = [hit["product_id"] for hit in hits[:10]]
        counts[qid] = sum(1 for pid in top10 if (qid, pid) in auto and (qid, pid) not in human)
    return counts


def affected_table(queries, spurious, containing, keyword_of, top10_auto):
    head = (
        r"\textbf{Mã} & \textbf{Truy vấn} & \textbf{Từ khoá bộ lọc} & \textbf{Loại} & "
        r"\textbf{Cặp bị chấm 0} & \textbf{Top-10 tự động} \\"
    )
    lines = [
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        r"\begin{longtable}{@{}r >{\raggedright\arraybackslash}p{5.2cm} >{\raggedright\arraybackslash}p{2.3cm} >{\raggedright\arraybackslash}p{3.0cm} >{\raggedleft\arraybackslash}p{1.4cm} >{\raggedleft\arraybackslash}p{1.5cm}@{}}",
        r"\caption{35 truy vấn bị đánh dấu khi đối soát bộ lọc chấm 0 tự động}\label{tab:pl-doi-soat}\\",
        r"\toprule",
        head,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        head,
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    rows = []
    for qid in sorted(set(spurious) | set(containing), key=int):
        kind = "accessory" if qid in containing else ("dress" if keyword_of[qid] == DRESS_KEYWORD else "spurious")
        pairs = containing[qid] if kind == "accessory" else spurious[qid]
        rows.append((qid, kind, pairs))
        lines.append(
            " & ".join(
                [
                    qid,
                    escape(queries[qid]["text"]),
                    escape(keyword_of[qid]),
                    FILTER_KIND[kind],
                    str(pairs),
                    f"{top10_auto[qid]}/10",
                ]
            )
            + r" \\"
        )
    lines += [r"\end{longtable}", r"\endgroup"]
    return "\n".join(lines), rows


def main():
    queries_list = audit.load_json(audit.EVAL / "queries.json")["queries"]
    queries = {str(q["id"]): q for q in queries_list}
    catalog = {p["product_id"]: p for p in audit.load_json(audit.ROOT / "data" / "processed" / "test.json")}
    pool = audit.load_json(audit.EVAL / "pool_top20.json")
    human, auto = audit.load_labels()
    names = audit.garment_names()
    spurious, containing, keyword_of = filter_statistics(queries, catalog, names, human, auto)
    top10_auto = lora_top10_auto(pool, human, auto)
    table, rows = affected_table(queries, spurious, containing, keyword_of, top10_auto)
    with open(QUERIES_OUTPUT, "w", encoding="utf-8") as handle:
        handle.write(queries_table(queries_list) + "\n")
    with open(AUDIT_OUTPUT, "w", encoding="utf-8") as handle:
        handle.write(table + "\n")
    for kind in FILTER_KIND:
        print(kind, sum(1 for _, k, _ in rows if k == kind), sum(p for _, k, p in rows if k == kind))
    by_kind = collections.Counter(kind for _, kind, _ in rows)
    print("queries", len(queries_list), collections.Counter(q["group"] for q in queries_list))
    print("affected", len(rows), dict(by_kind))
    print("spurious pairs", sum(spurious.values()), "queries", len(spurious))
    print("accessory pairs", sum(containing.values()), "queries", len(containing))
    print("overlap", sorted(set(spurious) & set(containing), key=int))
    print("accessory-only pair total in table", sum(p for _, k, p in rows if k == "accessory"))
    print("spurious-only pair total in table", sum(p for _, k, p in rows if k == "spurious"))


if __name__ == "__main__":
    main()
