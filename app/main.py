import os
import uuid
import httpx
from faster_whisper import WhisperModel
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()
app.mount("/static", StaticFiles(directory="/app/static"), name="static")
device = "cuda"

import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

print("載入 Whisper 模型中...")
model = WhisperModel("large-v3", device=device, compute_type="float16")
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
        segments, info = model.transcribe(
            upload_path,
            language="zh",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False
        )

        lines = []
        for segment in segments:
            start = int(segment.start)
            text = segment.text.strip()
            lines.append(f"[{start//60:02d}:{start%60:02d}] {text}")

        content = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        return JSONResponse({"status": "success", "text": content})

    finally:
        os.remove(upload_path)


OLLAMA_URL = "http://ollama:11434/api/generate"

DEFAULT_PROMPT_TEMPLATE = """請將以下會議逐字稿整理成正式會議記錄，格式如下：

---
# 會議記錄

## 一、討論重點
（條列本次會議主要討論的議題與內容摘要）

## 二、決議事項
（條列已確認的決定，每項標示編號）

## 三、追蹤事項
| 事項 | 負責人 | 期限 |
|------|--------|------|
| （待填） | （待填） | （待填） |

## 四、臨時動議
（條列臨時提出的事項；若無則填「無」）
- 下次會議時間：（若有提及請填入，否則填「待定」）

## 五、機密等級
（請依會議內容勾選）
- [ ] 一般
- [ ] 內部限閱
- [ ] 機密
---

注意事項：
- 必須使用繁體中文，以及臺灣常用的詞彙與表達方式
- 修正明顯的語音辨識錯誤
- 移除語氣詞（例如：嗯、啊、就是說）
- 若逐字稿中未提及某欄位的資訊，請如實標示「逐字稿中未提及」

逐字稿：
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
            "model": "qwen2.5:14b-instruct-q4_K_M",
            "prompt": prompt,
            "stream": False
        })

    resp_json = response.json()
    logging.debug(f"Ollama response: {resp_json}")
    if "response" not in resp_json:
        return JSONResponse({"status": "error", "detail": resp_json}, status_code=500)

    result = resp_json["response"]

    output_filename = file.filename.rsplit(".", 1)[0] + "_formatted.txt"
    output_path = f"/app/outputs/{output_filename}"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)

    return JSONResponse({"status": "success", "output": output_path, "formatted": result})
