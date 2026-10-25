import collections
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / "data" / "eval"
CHAPTERS = Path(__file__).resolve().parents[1] / "chapters"


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def escape(text):
    replacements = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "$": r"\$", "{": r"\{", "}": r"\}"}
    return "".join(replacements.get(ch, ch) for ch in text)


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


def queries_table(queries):
    lines = [
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.0}",
        r"\begin{longtable}{@{}r c p{6.4cm} r c p{6.4cm}@{}}",
        r"\caption{Toàn bộ 155 truy vấn của bộ đánh giá}\label{tab:pl_queries}\\",
        r"\toprule",
        r"\textbf{ID} & \textbf{N} & \textbf{Truy vấn} & \textbf{ID} & \textbf{N} & \textbf{Truy vấn} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\textbf{ID} & \textbf{N} & \textbf{Truy vấn} & \textbf{ID} & \textbf{N} & \textbf{Truy vấn} \\",
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


def spurious_table(queries):
    names = garment_names()
    auto = []
    for name in ("auto_zero_labels", "auto_zero_labels_ft"):
        auto += load_json(EVAL / f"{name}.json")
    per_query = collections.Counter()
    keyword_of = {}
    for row in auto:
        text = queries[row["query_id"]]["text"]
        keyword = keyword_in_query(text, names)
        if keyword.lower().split()[0] not in text.lower():
            per_query[row["query_id"]] += 1
            keyword_of[row["query_id"]] = keyword
    lines = [
        r"\begingroup",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{longtable}{@{}r p{7.0cm} p{3.6cm} r@{}}",
        r"\caption{21 truy vấn mà từ khoá do bộ lọc chọn không xuất hiện trong câu truy vấn}\label{tab:pl_spurious}\\",
        r"\toprule",
        r"\textbf{ID} & \textbf{Truy vấn} & \textbf{Từ khoá bộ lọc đã chọn} & \textbf{Cặp chấm 0 tự động} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\textbf{ID} & \textbf{Truy vấn} & \textbf{Từ khoá bộ lọc đã chọn} & \textbf{Cặp chấm 0 tự động} \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for qid in sorted(per_query, key=lambda x: int(x)):
        lines.append(
            f"{qid} & {escape(queries[qid]['text'])} & {escape(keyword_of[qid])} & {per_query[qid]}"
            + r" \\"
        )
    lines += [r"\end{longtable}", r"\endgroup"]
    return "\n".join(lines), len(per_query), sum(per_query.values())


def runs_table():
    import csv

    with open(ROOT / "experiments" / "runs.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Nhật ký thí nghiệm \texttt{experiments/runs.csv} (rút gọn các cột)}",
        r"\label{tab:pl_runs}",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\begin{tabular}{@{}l l c c c c r@{}}",
        r"\toprule",
        r"\textbf{run\_id} & \textbf{Ngày giờ} & \textbf{Hạng} & \textbf{Tốc độ học} & \textbf{Lô hiệu dụng} & \textbf{Epoch} & \textbf{Thời gian (s)} \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    r"\texttt{" + escape(row["run_id"]) + "}",
                    escape(row["date"]),
                    row["lora_rank"],
                    row["learning_rate"],
                    row["effective_batch"],
                    row["epochs"],
                    row["runtime_s"],
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    queries_list = load_json(EVAL / "queries.json")["queries"]
    queries = {str(q["id"]): q for q in queries_list}
    spurious, count, pairs = spurious_table(queries)
    content = "\n\n".join([queries_table(queries_list), spurious, runs_table()]) + "\n"
    with open(CHAPTERS / "phu_luc_bang.tex", "w", encoding="utf-8") as handle:
        handle.write(content)
    print(count, pairs)


if __name__ == "__main__":
    main()
