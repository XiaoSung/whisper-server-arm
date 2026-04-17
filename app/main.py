import os
import uuid
import httpx
import opencc
from faster_whisper import WhisperModel, BatchedInferencePipeline
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()
app.mount("/static", StaticFiles(directory="/app/static"), name="static")
device = "cuda"

import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

converter = opencc.OpenCC("s2twp")

print("載入 Whisper 模型中...")
_model = WhisperModel("large-v3", device=device, compute_type="float16")
model = BatchedInferencePipeline(model=_model)
print("模型載入完成！")

@app.get("/")
def index():
    return FileResponse("/app/static/index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    upload_path = f"/app/uploads/{job_id}_{file.filename}"
    output_path = f"/app/outputs/{job_id}.txt"

    with open(upload_path, "wb") as f:
        f.write(await file.read())

    try:
        segments, info = await run_in_threadpool(
            lambda: model.transcribe(
                upload_path,
                language="zh",
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
                batch_size=16
            )
        )

        lines = []
        for segment in segments:
            start = int(segment.start)
            text = converter.convert(segment.text.strip())
            lines.append(f"[{start//60:02d}:{start%60:02d}] {text}")

        content = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        return JSONResponse({"status": "success", "text": content})

    finally:
        os.remove(upload_path)


OLLAMA_URL = "http://ollama:11434/api/generate"

DEFAULT_PROMPT_TEMPLATE = """/think

你是一位資深會議秘書，擅長將冗長的逐字稿轉成精準完整的正式會議記錄。
你的產出將提交給未參與會議的主管，因此必須**自我包含**且**不可遺漏**任何被明確提及的事項。

# 步驟（請依序執行，不得跳過）

## Step 1：議題盤點（內部思考）
先通讀全文，列出所有被討論的議題（含主動議與被動提及），確認每個議題在後續輸出中都有對應段落。**寧可多列，不可漏列**。

## Step 2：分類判準
對每個發言片段，依下列定義分類：
- **討論重點**：雙方交換意見、資訊分享、狀況報告，但未拍板。
- **決議事項**：會中達成共識、簽核、正式通過的結論，通常有「決議」「通過」「同意」等語氣。
- **追蹤事項（Action Items）**：會中明確交辦的任務，記錄任務內容與期限（若有）。**不需填寫負責人，留白供使用者補填**。
- **臨時動議**：原議程外、會中臨時提出的提案，無論是否通過都需列出。**不需填寫提案人，留白供使用者補填**。
- **機密等級**：依內容敏感度標註為「公開 / 內部 / 機密 / 極機密」四級，並說明判定理由。

## Step 3：密度檢查
初稿完成後，回頭檢視每個議題，補齊以下遺漏的細節：
- 具體單位、數字、金額、日期、時程
- 爭議點與不同立場（以「有人主張…」「另一方認為…」表達，**不要試圖填入人名**）
- 擱置原因（若有議題未決）

# 人名處理原則（重要）
- **不得填寫任何人名**。逐字稿中的人名辨識不可靠，一律不抽取。
- 涉及發言者的段落，使用「與會者」「有成員」「某單位代表」等中性描述。
- 凡是需要人名的欄位（主席、出席者、負責人、提案人等），一律保留為空白 `___` 或 `（待填）`，供使用者自行補填。

# 輸出格式（嚴格遵守）

## 會議基本資訊
- 會議名稱：___
- 時間：___
- 地點：___
- 記錄：___

## 一、討論重點
依議題分段，每議題至少涵蓋：背景、各方立場、關鍵數據。
- 【議題名稱】
  - 背景說明：…
  - 討論內容：…（條列各方觀點，以中性詞代指發言者）

## 二、決議事項
編號列出，每項包含：決議內容、支持依據、生效時間。
1. …

## 三、追蹤事項
| 編號 | 任務內容 | 負責人 | 期限 | 備註 |
| --- | --- | --- | --- | --- |
| 1 | … | ___ | … | … |

## 四、臨時動議
1. 提案人：___ / 提案內容：… / 處理結果：…（通過/保留/否決）

## 五、機密等級
- 建議等級：[公開 / 內部 / 機密 / 極機密]
- 判定理由：…

# 硬性要求
- 逐字稿中**凡是有被討論超過兩次的主題**，必須獨立成一條討論重點，不得合併。
- 若某欄位在本次會議中無對應內容，請明確寫「本次會議無此項」，**不得省略該欄位**。
- 不得引入逐字稿以外的資訊；不得美化或推測未說明的細節。
- **嚴禁輸出人名**，即使逐字稿中明確出現，也以中性詞取代或留白。

# 逐字稿：
{transcript}

整理後："""

@app.get("/prompt-template")
def get_prompt_template():
    return JSONResponse({"template": DEFAULT_PROMPT_TEMPLATE})

@app.post("/format")
async def format_transcript(
    file: UploadFile = File(...),
    prompt_template: str = Form(None)
):
    content = (await file.read()).decode("utf-8")

    template = prompt_template if prompt_template else DEFAULT_PROMPT_TEMPLATE
    prompt = template.replace("{transcript}", content)

    async with httpx.AsyncClient(timeout=600) as client:
        response = await client.post(OLLAMA_URL, json={
            "model": "qwen3:32b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 65536,
                "num_predict": -1
            }
        })

    resp_json = response.json()
    logging.debug(f"Ollama response: {resp_json}")
    if "response" not in resp_json:
        return JSONResponse({"status": "error", "detail": resp_json}, status_code=500)

    result = converter.convert(resp_json["response"])

    output_filename = file.filename.rsplit(".", 1)[0] + "_formatted.txt"
    output_path = f"/app/outputs/{output_filename}"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)

    return JSONResponse({"status": "success", "output": output_path, "formatted": result})
