"""檢索：BM25、向量檢索、混合檢索（RRF）。

只用標準函式庫；向量檢索透過本機 Ollama 的 embedding API。
"""
import json
import math
import re
import urllib.request
from collections import Counter
from pathlib import Path

CJK = r"一-鿿"


def load_chunks(path="data/chunks.jsonl"):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ---------- 切詞 ----------

def tokenize(text):
    """英數字以單字為單位（保留小數），中文以兩字一組（bigram）。"""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text)
    for run in re.findall(f"[{CJK}]+", text):
        if len(run) == 1:
            tokens.append(run)
        tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


# ---------- 術語表：台灣繁體用語 → 文件用語 ----------

def load_glossary(path="glossary.json"):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def expand_query(query, glossary):
    """1. 附加英文術語（中文問、英文文件）；2. 整詞換成大陸用語；3. 逐字繁轉簡。"""
    english = [en for tw, en in glossary.get("en", {}).items() if tw in query]
    for tw, cn in sorted(glossary.get("terms", {}).items(), key=lambda kv: -len(kv[0])):
        query = query.replace(tw, cn)
    chars = glossary.get("chars", {})
    query = "".join(chars.get(ch, ch) for ch in query)
    return query + (" " + " ".join(english) if english else "")


# ---------- BM25 ----------

class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(c["text"])) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / len(self.lens)
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query, k=4):
        q = tokenize(query)
        scores = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in q:
                if t not in d:
                    continue
                tf = d[t]
                s += self.idf[t] * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [(self.chunks[i], s) for s, i in scores[:k]]


# ---------- 向量檢索（Ollama embedding） ----------

def ollama_embed(texts, model="bge-m3", host="http://localhost:11434"):
    req = urllib.request.Request(
        f"{host}/api/embed",
        data=json.dumps({"model": model, "input": texts}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["embeddings"]


def _norm(v):
    s = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / s for x in v]


class VectorIndex:
    """第一次建立時把所有段落的向量存到快取檔，之後直接讀取。"""

    def __init__(self, chunks, model="bge-m3", cache="data/embeddings.json"):
        self.chunks, self.model = chunks, model
        p = Path(cache)
        if p.exists():
            cached = json.loads(p.read_text())
            if cached.get("model") == model and len(cached["vectors"]) == len(chunks):
                self.vecs = cached["vectors"]
                return
        vecs = []
        for i in range(0, len(chunks), 16):
            vecs += ollama_embed([c["text"] for c in chunks[i:i + 16]], model)
        self.vecs = [_norm(v) for v in vecs]
        p.write_text(json.dumps({"model": model, "vectors": self.vecs}))

    def search(self, query, k=4):
        q = _norm(ollama_embed([query], self.model)[0])
        scored = sorted(((sum(a * b for a, b in zip(q, v)), i)
                         for i, v in enumerate(self.vecs)), reverse=True)
        return [(self.chunks[i], s) for s, i in scored[:k]]


# ---------- 混合檢索（Reciprocal Rank Fusion） ----------

def rrf(result_lists, k=4, c=60):
    score, keep = {}, {}
    for results in result_lists:
        for rank, (chunk, _) in enumerate(results):
            score[chunk["id"]] = score.get(chunk["id"], 0) + 1 / (c + rank + 1)
            keep[chunk["id"]] = chunk
    best = sorted(score, key=score.get, reverse=True)[:k]
    return [(keep[i], score[i]) for i in best]


class Retriever:
    """mode: bm25 | vector | hybrid；glossary=True 時先轉換查詢用語。"""

    def __init__(self, chunks, mode="bm25", glossary=None, embed_model="bge-m3"):
        self.mode, self.glossary = mode, glossary or {}
        self.bm25 = BM25(chunks) if mode in ("bm25", "hybrid") else None
        self.vec = VectorIndex(chunks, embed_model) if mode in ("vector", "hybrid") else None

    def search(self, query, k=4):
        q = expand_query(query, self.glossary) if self.glossary else query
        if self.mode == "bm25":
            return self.bm25.search(q, k)
        if self.mode == "vector":
            return self.vec.search(query, k)   # 向量模型本身能處理繁簡，用原始問題
        return rrf([self.bm25.search(q, k * 3), self.vec.search(query, k * 3)], k)
