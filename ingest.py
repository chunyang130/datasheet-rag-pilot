"""把技術規格書 PDF 切成段落，輸出 chunks.jsonl。

用法：
    python ingest.py --out data/chunks.jsonl

每份 PDF 依頁切開；單頁太長時再切成有重疊的小段。
段落編號格式：<doc>#p<頁碼>#<序號>，例如 esp32#p30#1。
"""
import argparse
import json
from pathlib import Path

import pdfplumber

DOCS = {
    "esp32": "data/esp32.pdf",
    "wroom": "data/wroom.pdf",
    "lm317": "data/lm317.pdf",
}
MAX_CHARS = 900   # 單段上限
OVERLAP = 200     # 相鄰兩段重疊的字數，避免表格被切斷


def split_page(text, max_chars=MAX_CHARS, overlap=OVERLAP):
    """依行累積成段，超過上限就切，下一段保留上一段結尾的重疊內容。"""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    chunks, cur = [], ""
    for ln in lines:
        if cur and len(cur) + len(ln) + 1 > max_chars:
            chunks.append(cur)
            cur = cur[-overlap:]
        cur = (cur + "\n" + ln) if cur else ln
    if cur:
        chunks.append(cur)
    return chunks


def ingest(docs=DOCS):
    rows = []
    for doc, path in docs.items():
        with pdfplumber.open(path) as pdf:
            for pno, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                for k, chunk in enumerate(split_page(text), 1):
                    rows.append({"id": f"{doc}#p{pno}#{k}", "doc": doc,
                                 "page": pno, "text": chunk})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/chunks.jsonl")
    args = ap.parse_args()
    rows = ingest()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_doc = {}
    for r in rows:
        by_doc[r["doc"]] = by_doc.get(r["doc"], 0) + 1
    print(f"共 {len(rows)} 段：{by_doc}")


if __name__ == "__main__":
    main()
