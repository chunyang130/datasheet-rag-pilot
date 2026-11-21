# 技術規格書 RAG 前導實驗：地端與雲端模型的可靠性比較

> 研究問題：技術文件不能離開公司、只能使用地端模型時，AI 能不能可靠地回答工程師的技術問題？

本實驗是碩士班研究的前導研究，延伸自 [course-qa-rag](https://github.com/chunyang130/course-qa-rag) 與 [model-benchmark](https://github.com/chunyang130/model-benchmark)，探討三個問題：

1. **檢索**：規格書含大量表格與數字，BM25、向量檢索、混合檢索哪種最準？
2. **驗證**：引用比對與數字檢查，能把錯誤回答降低多少？
3. **模型**：不同大小的地端模型，在正確率、速度與成本之間如何取捨？

## 知識庫

使用三份原廠公開規格書（版權屬原廠，**本 repo 不附 PDF**，請自行下載後放入 `data/`）：

| 檔名 | 文件 | 語言 | 頁數 | 下載 |
| --- | --- | --- | --- | --- |
| `data/esp32.pdf` | ESP32 系列芯片技術規格書 v5.3 | 簡體中文 | 74 | documentation.espressif.com/esp32_datasheet_cn.pdf |
| `data/wroom.pdf` | ESP32-WROOM-32E/32UE 技術規格書 v2.1 | 簡體中文 | 52 | documentation.espressif.com/esp32-wroom-32e_esp32-wroom-32ue_datasheet_cn.pdf |
| `data/lm317.pdf` | LM317 datasheet, SLVS044Z (Apr 2025) | 英文 | 45 | ti.com/product/LM317 |

## 題組（`eval/questions.json`，30 題，以繁體中文提問）

| 題型 | 題數 | 測試重點 |
| --- | --- | --- |
| 查表（lookup） | 13 | 從表格找出單一數值 |
| 但書（caveat） | 9 | 答案有版本或條件差異，漏講即算錯 |
| 混淆（confusion） | 4 | 晶片與模組規格書數值不同，必須分清楚 |
| 計算（calc） | 1 | 套用文件公式 |
| 拒答（refuse） | 3 | 文件中沒有答案 |

每題標註來源文件與頁碼；12 題 LM317 題目是「中文問、英文文件」的跨語言題。

## 流程

1. **切段**（`ingest.py`）：以 pdfplumber 逐頁抽取文字，單頁過長再切成有 200 字重疊的段落，共 282 段。
2. **檢索**（`retrieval.py`）：
   - BM25：英數字以單字切詞，中文兩字一組。
   - 術語表（`glossary.json`，選用）：繁轉簡、台灣用語轉大陸用語（晶片→芯片），並附加英文術語。
   - 向量檢索：本機 Ollama 的 `bge-m3` 多語 embedding。
   - 混合檢索：BM25 與向量結果以 RRF 合併。
3. **生成**（`rag.py`）：只把前 4 段交給模型，要求輸出 JSON（能否回答、答案、引用）。
4. **驗證**（可關閉）：引用原文必須逐字出現在段落中；答案中的每個數字必須出現在引用段落或題目中。未通過則回饋錯誤、重試一次。
5. **評分**：答案須包含所有關鍵數值（但書題須列出所有條件的數值）；拒答題須回答找不到。

## 結果（2026-11-21，Colab T4 GPU）

### 一句話結論

**檢索是瓶頸，雲端模型的優勢在「沒把握時會說找不到」。** Claude Haiku 30 題答對 24 題；地端 Qwen 7B 答對 17 題，而且答錯時多半很有自信地給出錯誤數字。

### 1. 檢索（27 題有標準頁碼，命中＝前 4 段中有正確頁面）

| 設定 | 命中 | 查表 | 但書 | 混淆 | 計算 |
| --- | --- | --- | --- | --- | --- |
| BM25 | 12/27 | 6/13 | 4/9 | 2/4 | 0/1 |
| 向量（bge-m3，不需術語表） | 18/27 | 8/13 | 7/9 | 2/4 | 1/1 |
| 混合＋術語表 | 19/27 | 9/13 | 7/9 | 3/4 | 0/1 |
| BM25＋術語表 | 21/27 | 10/13 | 8/9 | 3/4 | 0/1 |

- 只用 BM25 時，中文問題對不上英文的 LM317 文件，繁體問題也對不上簡體的 ESP32 文件。
- 向量檢索不需要人工術語表，就從 12 題提升到 18 題。
- BM25＋術語表最高，但**術語表是看到錯題後才建立**，有針對本題組修正的成分；混合檢索並沒有比單用 BM25＋術語表更好。

### 2. 生成（混合檢索＋術語表，30 題）

| 模型 | 驗證 | 整體 | 查表 | 但書 | 混淆 | 計算 | 拒答 | 每題秒數 | 平均 tokens（入/出） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Claude Haiku 5.5 | 關 | **24/30** | 11/13 | **6/9** | 3/4 | 1/1 | 3/3 | 3.2 | 2490/671 |
| Claude Haiku 5.5 | 開 | **24/30** | 11/13 | **6/9** | 3/4 | 1/1 | 3/3 | 3.5 | 2742/828 |
| Qwen2.5 7B（地端） | 關 | 17/30 | 10/13 | 3/9 | 1/4 | 0/1 | 3/3 | 3.6 | 1822/140 |
| Qwen2.5 7B（地端） | 開 | 17/30 | 10/13 | 3/9 | 1/4 | 0/1 | 3/3 | 5.8 | 2344/175 |
| Qwen2.5 3B（地端） | 關 | 16/30 | 11/13 | 1/9 | 1/4 | 0/1 | 3/3 | 1.7 | 1822/117 |
| Qwen2.5 3B（地端） | 開 | 15/30 | 10/13 | 1/9 | 1/4 | 0/1 | 3/3 | 2.8 | 2541/1535 |

### 3. 答錯的方式（不含拒答題的 27 題）

| 模型（驗證開） | 檢索命中時答對 | 檢索沒命中時答對 | 答錯但給了答案 | 答錯但承認找不到 |
| --- | --- | --- | --- | --- |
| Claude Haiku 5.5 | 18/19 | 3/8 | 2 | 4 |
| Qwen2.5 7B | 13/19 | 1/8 | 8 | 5 |
| Qwen2.5 3B | 10/19 | 2/8 | 8 | 5 |

### 發現

1. **檢索是最大的瓶頸**：Claude 在檢索命中的 19 題答對 18 題；它答錯的 6 題中，5 題是檢索沒找到正確頁面。
2. **地端模型的危險在「自信地答錯」**：問模組的儲存溫度（−40～105 °C）時，檢索沒找到正確頁面，Claude 回答「找不到」，Qwen 3B 與 7B 都答成工作溫度（−40～85 °C）。
3. **但書題最能拉開差距**：LM317 熱阻依晶片版本有兩個數值，Claude 兩個都列出，Qwen 只列一個。
4. **數字檢查擋不住「用錯地方的數字」**：85 °C 確實出現在引用段落，所以通過檢查。驗證開與關的正確率幾乎相同，只比對數字不夠，下一步要檢查數字與問題的對應關係。
5. **費用**：Claude Haiku 驗證開時平均每題 2,742／828 tokens；以 2026 年 11 月查詢的牌價（每百萬 tokens 輸入 0.10、輸出 0.50 美元，以官網為準）換算，每千題約 0.7 美元。地端模型沒有 API 費用，但需要 GPU。

### 修正紀錄

第一輪 Claude 的輸出上限設為 800 tokens，約一半的題目答到一半被截斷，或在內部思考時就用完額度，結果只有 13～15/30。上限改為 4000，並在截斷時明確標示後重跑。第一輪原始結果保留在 `results/round1_maxtok800_*.json`。

## PDF 抽取時發現的問題

| 問題 | 例子 | 影響 |
| --- | --- | --- |
| 常見工具抽不出中文 | pdftotext 讀 Espressif 中文規格書時，中文字全部消失 | 改用 pdfplumber |
| 空格消失 | 「448KB ROM」變成「448KBROM」 | 影響切詞與檢索 |
| 符號錯誤 | 「Ω」被抽成「W」；下標被拆開（V IN） | 影響數字與單位的驗證 |

## 執行方式

```powershell
pip install -r requirements.txt
python ingest.py                                   # 切段（需先放好 3 份 PDF）
python eval_retrieval.py --mode bm25 --glossary    # 檢索評估（免費）
python rag.py run --provider mock                  # 完整流程示範（免費）
python -m unittest test_rag.py                     # 自動測試（14 項）
```

**雲端模型**（需要 Anthropic API 金鑰，只從環境變數讀取）：

```powershell
$env:ANTHROPIC_API_KEY="你的金鑰"
python rag.py run --provider claude --model claude-haiku-5-5 --glossary
python rag.py run --provider claude --model claude-haiku-5-5 --glossary --no-verify
```

**地端模型與向量檢索**（需要 [Ollama](https://ollama.com)；電腦跑不動時可用 `colab.ipynb`）：

```powershell
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
ollama pull bge-m3
python eval_retrieval.py --mode vector
python eval_retrieval.py --mode hybrid --glossary
python rag.py run --provider ollama --model qwen2.5:7b --retriever hybrid --glossary --repeat 3
```

**比較表**：`python rag.py report results/*.json`

## 實驗設計

| 變因 | 設定 |
| --- | --- |
| 檢索 | BM25、BM25＋術語表、向量、混合 |
| 驗證 | 開、關 |
| 模型 | Claude Haiku（雲端）、Qwen2.5 3B、Qwen2.5 7B（地端） |
| 重複 | 本輪每個設定只跑 1 次；下一輪預計每個設定跑 3 次，觀察結果是否穩定 |

## 限制

- 只有 30 題、3 份文件，結論只適用於本題組。
- 術語表為事後建立，需要用未看過的新題目重新評估。
- 計算題的答案（13.75 V）不在原文中，會被數字檢查標記；這是目前驗證規則的已知限制。
- 關鍵字評分只檢查數值是否出現，無法判斷說明是否正確。

## 檔案說明

| 檔案 | 內容 |
| --- | --- |
| `ingest.py` | PDF 逐頁抽取與切段 |
| `retrieval.py` | BM25、術語表、向量檢索、RRF 混合檢索 |
| `eval_retrieval.py` | 檢索命中率評估 |
| `rag.py` | 生成、驗證、重試、評分與比較表 |
| `results/` | 所有實驗的原始結果與比較表 |
| `glossary.json` | 繁簡與中英術語表 |
| `eval/questions.json` | 30 題題組，附文件、頁碼、答案與關鍵字 |
| `eval/mock_responses.json` | 示範用模擬回覆（q07 第一次故意答錯數字，展示驗證與重試） |
| `test_rag.py` | 自動測試（14 項） |
| `colab.ipynb` | 在 Google Colab 執行地端模型 |
