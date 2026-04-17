# 妙妙小工具之會議記錄系統

以 Faster-Whisper + Ollama 為核心的本地端會議記錄自動化工具。上傳音檔後自動轉錄為逐字稿，再透過大型語言模型整理成結構化的正式會議記錄，全程在本機 GPU 執行，不需要任何雲端服務。

## 功能概覽

- **語音轉逐字稿**：使用 `faster-whisper large-v3` 模型（CUDA 加速），支援 mp3、mp4、m4a、wav、flac 等格式
- **逐字稿整理**：透過 Ollama（`qwen3:32b`）將逐字稿整理成包含討論重點、決議事項、追蹤事項的正式會議記錄
- **繁體中文輸出**：整合 OpenCC，自動將簡體轉換為繁體中文（臺灣用語）
- **自訂 Prompt**：內建預設會議記錄 Prompt，可透過 Web UI 自由修改
- **網頁操作介面**：三步驟引導式 UI，無需命令列操作

## 系統架構

```
[瀏覽器 Web UI]
       │
       ├──(音檔)──► FastAPI /transcribe ──► Faster-Whisper (GPU)
       │
       └──(逐字稿)──► FastAPI /format ──► Ollama qwen3:32b (GPU)
```

## 環境需求

- **NVIDIA GPU**（支援 CUDA）
- Docker + Docker Compose
- NVIDIA Container Toolkit（`nvidia-docker`）

> 映像檔基底為 `nvcr.io/nvidia/pytorch:25.01-py3`，支援 GB10（Grace Blackwell / DGX Spark）架構。

## 快速開始

```bash
# 複製專案
git clone <repo-url>
cd whisper-server-arm

# 建置並啟動（首次需下載模型，約需數分鐘）
docker compose up --build
```

服務啟動後，開啟瀏覽器前往 `http://localhost:8000`。

## 使用方式

1. **Step 1（選填）** — 上傳音檔，點選「開始轉錄」，等待 Whisper 產出逐字稿
2. **Step 2** — 確認或直接貼上逐字稿內容
3. **Step 2.5** — 視需要修改整理 Prompt（預設為會議秘書角色）
4. 點選「整理成會議記錄」，等待 LLM 完成（視逐字稿長度，約需 1–5 分鐘）
5. **Step 3** — 檢閱結果並下載 `.txt` 檔案

## API 端點

| 方法 | 路徑 | 說明 |
|------|------|------|
| `GET` | `/` | Web UI |
| `GET` | `/health` | 健康檢查 |
| `GET` | `/prompt-template` | 取得預設 Prompt 範本 |
| `POST` | `/transcribe` | 上傳音檔，回傳帶時間戳記的逐字稿 |
| `POST` | `/format` | 上傳逐字稿文字，回傳整理後的會議記錄 |

### `/transcribe` 回傳格式

```json
{
  "status": "success",
  "text": "[00:00] 今天討論的第一個議題...\n[01:23] ..."
}
```

### `/format` 請求參數

| 欄位 | 型別 | 說明 |
|------|------|------|
| `file` | `multipart/form-data` | 逐字稿 `.txt` 檔案 |
| `prompt_template` | `string`（選填） | 自訂 Prompt，需包含 `{transcript}` 佔位符 |

## 預設 Prompt 行為

預設 Prompt 要求 LLM 以資深會議秘書身份，依序輸出：

1. **會議基本資訊**（名稱、時間、地點，留白供人工補填）
2. **討論重點**（依議題分段）
3. **決議事項**（編號列出）
4. **追蹤事項**（表格，負責人欄留白）
5. **臨時動議**
6. **機密等級建議**

所有人名欄位一律留白，避免語音辨識錯誤造成誤植。

## 專案結構

```
.
├── Dockerfile
├── docker-compose.yml
└── app/
    ├── main.py          # FastAPI 應用程式
    └── static/
        └── index.html   # 前端 Web UI
```

## 資料儲存

| 路徑 | 說明 |
|------|------|
| `./uploads/` | 上傳的音檔暫存（轉錄完成後自動刪除） |
| `./outputs/` | 轉錄結果與會議記錄輸出檔案 |
| Docker Volume `whisper_model_cache` | Hugging Face 模型快取 |
| Docker Volume `ollama_data` | Ollama 模型資料 |
