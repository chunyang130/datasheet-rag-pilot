"""只評估檢索（免費，不呼叫任何生成模型）。

命中定義：前 k 段中，至少有一段來自標準答案的「文件＋頁碼」。
拒答題沒有標準頁碼，不列入命中率。

用法：
    python eval_retrieval.py --mode bm25
    python eval_retrieval.py --mode bm25 --glossary
    python eval_retrieval.py --mode hybrid --glossary     # 需要本機 Ollama
"""
import argparse
import json
from collections import defaultdict

from retrieval import Retriever, load_chunks, load_glossary


def evaluate(mode="bm25", use_glossary=False, k=4, questions="eval/questions.json"):
    qs = json.load(open(questions, encoding="utf-8"))["questions"]
    ret = Retriever(load_chunks(), mode, load_glossary() if use_glossary else None)
    rows, by_type = [], defaultdict(lambda: [0, 0])
    for q in qs:
        if q["page"] is None:
            continue
        hits = ret.search(q["question"], k)
        ids = [c["id"] for c, _ in hits]
        rank = next((i + 1 for i, (c, _) in enumerate(hits)
                     if c["doc"] == q["doc"] and c["page"] == q["page"]), None)
        rows.append({"id": q["id"], "type": q["type"], "gold": f'{q["doc"]}#p{q["page"]}',
                     "rank": rank, "top": ids})
        by_type[q["type"]][0] += rank is not None
        by_type[q["type"]][1] += 1
    return rows, dict(by_type)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="bm25", choices=["bm25", "vector", "hybrid"])
    ap.add_argument("--glossary", action="store_true")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out")
    a = ap.parse_args()
    rows, by_type = evaluate(a.mode, a.glossary, a.k)
    hit = sum(r["rank"] is not None for r in rows)
    label = a.mode + ("+glossary" if a.glossary else "")
    print(f"[{label}] 命中 {hit}/{len(rows)}（前 {a.k} 段）")
    for t, (h, n) in by_type.items():
        print(f"  {t:10s} {h}/{n}")
    for r in rows:
        mark = f"第{r['rank']}名" if r["rank"] else "未命中"
        print(f"  {r['id']} {r['gold']:12s} {mark:6s} {', '.join(r['top'])}")
    if a.out:
        json.dump({"label": label, "rows": rows, "by_type": by_type},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
