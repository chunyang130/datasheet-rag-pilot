"""自動測試：python -m unittest test_rag.py（不需 PDF、不呼叫任何模型）"""
import unittest

from ingest import split_page
from rag import score, verify
from retrieval import BM25, expand_query, rrf, tokenize

CHUNKS = [
    {"id": "lm317#p7#1", "doc": "lm317", "page": 7,
     "text": "R Junction-to-ambient thermal resistance 66.8 23.5 °C/W"},
    {"id": "esp32#p30#2", "doc": "esp32", "page": 30,
     "text": "Hibernation 仅有RTC定时器处于工作状态 5µA"},
]
HITS = [(c, 1.0) for c in CHUNKS]


class TestTokenize(unittest.TestCase):
    def test_mixed(self):
        t = tokenize("ESP32 晶片 1.25V")
        self.assertIn("esp32", t)
        self.assertIn("1.25v", t) if "1.25v" in t else self.assertIn("1.25", t)
        self.assertIn("晶片", t)

    def test_bigram(self):
        self.assertEqual(tokenize("存储温度"), ["存储", "储温", "温度"])


class TestGlossary(unittest.TestCase):
    G = {"terms": {"晶片": "芯片"}, "en": {"熱阻": "thermal resistance"},
         "chars": {"溫": "温", "儲": "储"}}

    def test_convert_and_append(self):
        q = expand_query("晶片的熱阻與儲存溫度", self.G)
        self.assertIn("芯片", q)
        self.assertIn("储存温度", q)
        self.assertTrue(q.endswith("thermal resistance"))


class TestRetrieval(unittest.TestCase):
    def test_bm25_finds_page(self):
        top = BM25(CHUNKS).search("thermal resistance", 1)
        self.assertEqual(top[0][0]["id"], "lm317#p7#1")

    def test_rrf_merges(self):
        a = [(CHUNKS[0], 1), (CHUNKS[1], 0.5)]
        b = [(CHUNKS[1], 1), (CHUNKS[0], 0.5)]
        self.assertEqual(len(rrf([a, b], k=2)), 2)


class TestVerify(unittest.TestCase):
    def test_pass(self):
        out = {"answerable": True, "answer": "66.8 °C/W",
               "citations": [{"id": "lm317#p7#1", "quote": "thermal resistance 66.8"}]}
        self.assertEqual(verify(out, HITS, "熱阻？"), [])

    def test_wrong_number(self):
        out = {"answerable": True, "answer": "66.0 °C/W",
               "citations": [{"id": "lm317#p7#1", "quote": "thermal resistance 66.8"}]}
        self.assertTrue(any("數字" in e for e in verify(out, HITS, "熱阻？")))

    def test_fake_quote(self):
        out = {"answerable": True, "answer": "5µA",
               "citations": [{"id": "esp32#p30#2", "quote": "功耗为5µA"}]}
        self.assertTrue(any("引用原文" in e for e in verify(out, HITS, "?")))

    def test_unknown_chunk(self):
        out = {"answerable": True, "answer": "5",
               "citations": [{"id": "esp32#p99#1", "quote": "x"}]}
        self.assertTrue(any("不在提供的段落" in e for e in verify(out, HITS, "?")))

    def test_number_from_question_allowed(self):
        out = {"answerable": True, "answer": "R1 = 240 Ω 時 66.8",
               "citations": [{"id": "lm317#p7#1", "quote": "66.8"}]}
        self.assertEqual(verify(out, HITS, "R1 = 240 Ω"), [])

    def test_refusal_needs_no_citation(self):
        self.assertEqual(verify({"answerable": False, "answer": "文件中找不到"}, HITS, "?"), [])


class TestScore(unittest.TestCase):
    def test_caveat_needs_all_keywords(self):
        q = {"type": "caveat", "keywords": ["66.8", "59.6"]}
        self.assertFalse(score(q, {"answerable": True, "answer": "66.8 °C/W"}))
        self.assertTrue(score(q, {"answerable": True, "answer": "舊版 66.8、新版 59.6"}))

    def test_refuse(self):
        q = {"type": "refuse", "keywords": []}
        self.assertTrue(score(q, {"answerable": False, "answer": "文件中找不到"}))
        self.assertFalse(score(q, {"answerable": True, "answer": "有"}))


class TestSplit(unittest.TestCase):
    def test_overlap(self):
        text = "\n".join(f"line{i:03d} " + "x" * 50 for i in range(60))
        parts = split_page(text, max_chars=900, overlap=200)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(p) <= 900 + 60 for p in parts))


if __name__ == "__main__":
    unittest.main()
