"""技術文件 RAG：檢索 → 生成 → 程式驗證 → 評分。

用法（範例）：
    # 免費示範：用預先寫好的回覆跑完整流程
    python rag.py run --provider mock --mock eval/mock_responses.json

    # 雲端模型（需要 ANTHROPIC_API_KEY）
    python rag.py run --provider claude --model claude-haiku-5-5 --glossary

    # 地端模型（需要本機 Ollama）
    python rag.py run --provider ollama --model qwen2.5:7b --glossary --repeat 3

    # 比較所有結果
    python rag.py report results/*.json

實驗變因：
    --retriever bm25 | vector | hybrid    研究問題 1：哪種檢索最準
    --verify / --no-verify                研究問題 2：程式檢查能降低多少錯誤
    --provider + --model                  研究問題 3：模型大小與正確率、速度、成本
"""
import argparse
import glob
import json
import os
import re
import statistics
import time
import urllib.request
from pathlib import Path

from retrieval import Retriever, load_chunks, load_glossary

PROMPT = """你是技術文件問答助理。只能根據下面提供的文件段落回答，不可使用段落以外的知識。

規則：
1. 段落中找不到答案時，answerable 填 false，answer 填「文件中找不到」。
2. 答案若有條件或版本差異（例如不同型號、不同晶片版本、不同測試條件），必須全部列出。
3. 每個答案都要附引用：段落編號，以及從該段落逐字複製的原文片段。
4. 以繁體中文回答，數值保留原文單位。
5. 只輸出 JSON，格式：
{{"answerable": true, "answer": "...", "citations": [{{"id": "段落編號", "quote": "逐字原文"}}]}}

文件段落：
{context}

問題：{question}"""

RETRY = "\n\n你上一次的回答沒有通過檢查：{errors}\n請修正後重新輸出 JSON。"


# ---------- 模型呼叫 ----------

def _post(url, payload, headers, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def call_claude(prompt, model):
    data = _post("https://api.anthropic.com/v1/messages",
                 {"model": model, "max_tokens": 800, "temperature": 0,
                  "messages": [{"role": "user", "content": prompt}]},
                 {"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                  "anthropic-version": "2023-06-01"})
    text = "".join(b.get("text", "") for b in data["content"])
    u = data.get("usage", {})
    return text, u.get("input_tokens", 0), u.get("output_tokens", 0)


def call_ollama(prompt, model, host="http://localhost:11434"):
    data = _post(f"{host}/api/chat",
                 {"model": model, "stream": False, "format": "json",
                  "options": {"temperature": 0},
                  "messages": [{"role": "user", "content": prompt}]}, {}, timeout=900)
    return (data["message"]["content"], data.get("prompt_eval_count", 0),
            data.get("eval_count", 0))


class Mock:
    def __init__(self, path):
        self.responses = json.load(open(path, encoding="utf-8"))
        self.seen = {}

    def __call__(self, qid):
        n = self.seen.get(qid, 0)
        self.seen[qid] = n + 1
        options = self.responses.get(qid, ['{"answerable": false, "answer": "文件中找不到", "citations": []}'])
        return options[min(n, len(options) - 1)], 0, 0


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


# ---------- 程式驗證 ----------

def _squash(s):
    return re.sub(r"\s+", "", s).replace("−", "-").replace("–", "-")


def numbers(s):
    return set(re.findall(r"\d+(?:\.\d+)?", s))


def verify(out, hits, question):
    """回傳錯誤清單；空清單代表通過。"""
    if out is None:
        return ["輸出不是合法 JSON"]
    if not out.get("answerable"):
        return []
    by_id = {c["id"]: c for c, _ in hits}
    cites = out.get("citations") or []
    if not cites:
        return ["可回答的題目必須附引用"]
    errors, cited_text = [], ""
    for c in cites:
        chunk = by_id.get(c.get("id"))
        if chunk is None:
            errors.append(f"引用的段落 {c.get('id')} 不在提供的段落中")
            continue
        cited_text += chunk["text"]
        if _squash(c.get("quote", "")) not in _squash(chunk["text"]):
            errors.append(f"引用原文與段落 {c['id']} 不符")
    # 數字從原文抽取，不先移除空白，避免相鄰數字黏在一起（66.8 23.5 → 66.823.5）
    allowed = numbers(cited_text) | numbers(question)
    missing = sorted(numbers(out.get("answer", "")) - allowed)
    if missing:
        errors.append(f"答案中的數字 {missing} 不在引用段落中")
    return errors


# ---------- 評分 ----------

def score(q, out):
    if out is None:
        return False
    if q["type"] == "refuse":
        return not out.get("answerable")
    ans = _squash(out.get("answer", "")).lower()
    return bool(out.get("answerable")) and all(_squash(k).lower() in ans for k in q["keywords"])


# ---------- 執行一輪 ----------

def run(args):
    qs = json.load(open(args.questions, encoding="utf-8"))["questions"]
    ret = Retriever(load_chunks(), args.retriever, load_glossary() if args.glossary else None)
    mock = Mock(args.mock) if args.provider == "mock" else None
    label = "_".join([args.provider, args.model.replace(":", "-").replace("/", "-"),
                      args.retriever + ("+gl" if args.glossary else ""),
                      "verify" if args.verify else "noverify"])
    rows = []
    for rep in range(args.repeat):
        for q in qs:
            hits = ret.search(q["question"], args.k)
            context = "\n\n".join(f"[{c['id']}]\n{c['text']}" for c, _ in hits)
            prompt = PROMPT.format(context=context, question=q["question"])
            t0, tin, tout, attempts, errors = time.time(), 0, 0, 0, []
            while True:
                attempts += 1
                try:
                    if mock:
                        text, i, o = mock(q["id"])
                    elif args.provider == "claude":
                        text, i, o = call_claude(prompt, args.model)
                    else:
                        text, i, o = call_ollama(prompt, args.model)
                except Exception as e:          # 單題失敗不中斷整輪
                    text, i, o = "", 0, 0
                    errors = [f"API 失敗：{e}"]
                    break
                tin, tout = tin + i, tout + o
                out = parse_json(text)
                errors = verify(out, hits, q["question"]) if args.verify else []
                if not errors or attempts >= 2:
                    break
                prompt += RETRY.format(errors="；".join(errors))
            out = parse_json(text)
            gold_hit = q["page"] is not None and any(
                c["doc"] == q["doc"] and c["page"] == q["page"] for c, _ in hits)
            rows.append({"rep": rep, "id": q["id"], "type": q["type"],
                         "correct": score(q, out), "retrieval_hit": gold_hit,
                         "attempts": attempts, "verify_errors": errors,
                         "seconds": round(time.time() - t0, 2),
                         "tokens_in": tin, "tokens_out": tout,
                         "answer": (out or {}).get("answer"), "raw": text})
            print(f"[{rep}] {q['id']} {'對' if rows[-1]['correct'] else '錯'} "
                  f"重試{attempts - 1} {rows[-1]['seconds']}s")
    Path("results").mkdir(exist_ok=True)
    path = f"results/{label}.json"
    json.dump({"label": label, "args": vars(args), "rows": rows},
              open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"已存檔：{path}")
    summarize([path])


# ---------- 比較報表 ----------

def summarize(paths):
    print("| 設定 | 整體答對 | 查表 | 但書 | 混淆 | 計算 | 拒答 | 觸發重試 | 每題秒數中位數 | 平均 tokens（入/出）|")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        rows = d["rows"]

        def rate(t=None):
            r = [x for x in rows if t is None or x["type"] == t]
            return f"{sum(x['correct'] for x in r)}/{len(r)}" if r else "-"
        retried = sum(x["attempts"] > 1 for x in rows)
        med = statistics.median(x["seconds"] for x in rows)
        tin = sum(x["tokens_in"] for x in rows) / len(rows)
        tout = sum(x["tokens_out"] for x in rows) / len(rows)
        print(f"| {d['label']} | {rate()} | {rate('lookup')} | {rate('caveat')} | "
              f"{rate('confusion')} | {rate('calc')} | {rate('refuse')} | {retried} | "
              f"{med:.2f} | {tin:.0f}/{tout:.0f} |")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--provider", choices=["mock", "claude", "ollama"], default="mock")
    r.add_argument("--model", default="mock")
    r.add_argument("--mock", default="eval/mock_responses.json")
    r.add_argument("--retriever", choices=["bm25", "vector", "hybrid"], default="bm25")
    r.add_argument("--glossary", action="store_true")
    r.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    r.add_argument("--k", type=int, default=4)
    r.add_argument("--repeat", type=int, default=1)
    r.add_argument("--questions", default="eval/questions.json")
    s = sub.add_parser("report")
    s.add_argument("paths", nargs="+")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    else:
        summarize(sorted(set(sum((glob.glob(p) for p in a.paths), []))))


if __name__ == "__main__":
    main()
