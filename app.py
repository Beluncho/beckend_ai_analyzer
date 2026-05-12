import os
import re
import json
import tempfile
import subprocess
import logging
from typing import Optional, List, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(encoding="utf-8")
# ========== НАСТРОЙКИ ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

app = FastAPI(title="AI Call Analyzer", version="1.0.0")

# ========== CORS ==========
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.(lovableproject\.com|lovable\.app|onrender\.com)",
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

# ========== КЛЮЧ ИЗ ОКРУЖЕНИЯ ==========
PROXY_API_KEY = os.getenv("PROXY_API_KEY")

if not PROXY_API_KEY:
    raise RuntimeError("PROXY_API_KEY is not set in environment variables")

# Клиент OpenAI через прокси
_openai_client = OpenAI(
    api_key=PROXY_API_KEY,
    base_url="https://openai.api.proxyapi.ru/v1",
    timeout=120.0,
    max_retries=2,
)


# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========
def normalize_criteria(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
        except Exception:
            pass
        parts = re.split(r"[\n;]+", s)
        return [p.strip() for p in parts if p.strip()]
    return [str(raw).strip()] if str(raw).strip() else []


def ffmpeg_to_wav(src_path: str, dst_path: str) -> None:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", src_path,
        "-ac", "1", "-ar", "16000",
        dst_path
    ]
    subprocess.check_call(cmd)


def _extract_text_from_transcription(resp: Any) -> str:
    if isinstance(resp, str):
        return resp.strip()
    txt = getattr(resp, "text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()
    return str(resp).strip()


def transcribe_audio(client: OpenAI, wav_path: str) -> str:
    # Используем модели из документации без префикса
    model_candidates = ["whisper-1", "gpt-4o-mini-transcribe"]

    for m in model_candidates:
        try:
            logging.info(f"Attempting transcription with model: {m}")
            with open(wav_path, "rb") as f:
                resp = client.audio.transcriptions.create(
                    model=m,
                    file=f,
                    response_format="text",
                    language="ru",  # Явно указываем русский язык
                )
            text = _extract_text_from_transcription(resp)
            if text:
                logging.info(f"Transcription successful with {m}, length: {len(text)}")
                return text
            else:
                logging.warning(f"Empty transcription with {m}")
        except Exception as e:
            logging.error(f"Transcription failed with {m}: {type(e).__name__}: {e}")
            continue

    raise RuntimeError("All transcription models failed")


def diarize_text(client: OpenAI, raw_transcript: str) -> str:
    # Используем модели из документации
    model_candidates = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"]

    for m in model_candidates:
        try:
            logging.info(f"Attempting diarization with model: {m}")
            resp = client.chat.completions.create(
                model=m,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ты форматировщик расшифровок звонков.\n"
                            "Разбей текст на реплики и поставь метки 'Спикер 1:', 'Спикер 2:'.\n"
                            "Не меняй слова. Только метки."
                        ),
                    },
                    {"role": "user", "content": raw_transcript},
                ],
            )
            out = resp.choices[0].message.content.strip()
            if out:
                logging.info(f"Diarization successful with {m}")
                return out
        except Exception as e:
            logging.error(f"Diarization failed with {m}: {type(e).__name__}: {e}")
            continue

    # Fallback
    logging.warning("Using fallback diarization")
    sentences = [s.strip() for s in re.split(r"(?<=[\.\!\?\n])\s+", raw_transcript.strip()) if s.strip()]
    lines = []
    speaker = 1
    for s in sentences:
        lines.append(f"Спикер {speaker}: {s}")
        speaker = 2 if speaker == 1 else 1
    return "\n".join(lines)


def analyze_dialogue(client: OpenAI, dialogue_text: str, criteria: List[str]) -> str:
    # Используем модели из документации
    model_candidates = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"]
    criteria_block = "\n".join([f"- {c}" for c in criteria]) if criteria else "- (критерии не переданы)"

    system_prompt = (
        "Ты эксперт по анализу диалогов.\n"
        "Проанализируй диалог по заданным критериям.\n"
        "Верни разбор каждого критерия + общий анализ: сильные стороны, слабые места, рекомендации.\n"
        "Пиши на русском."
    )

    user_prompt = f"Критерии:\n{criteria_block}\n\nДиалог:\n{dialogue_text}"

    for m in model_candidates:
        try:
            logging.info(f"Attempting analysis with model: {m}")
            resp = client.chat.completions.create(
                model=m,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            out = resp.choices[0].message.content.strip()
            if out:
                logging.info(f"Analysis successful with {m}")
                return out
        except Exception as e:
            logging.error(f"Analysis failed with {m}: {type(e).__name__}: {e}")
            continue

    raise RuntimeError("All analysis models failed")


# ========== ТЕСТОВЫЙ ЭНДПОИНТ ДЛЯ ДИАГНОСТИКИ ==========
@app.get("/test")
async def test():
    """Тестовый эндпоинт для проверки подключения к API"""
    results = {}

    # Тест 1: Список моделей
    try:
        models = _openai_client.models.list()
        results["models_list"] = "success (got models)"
    except Exception as e:
        results["models_list"] = f"failed: {str(e)}"

    # Тест 2: Простой chat completion
    try:
        test_response = _openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'API works'"}],
            max_tokens=10
        )
        results["chat_test"] = f"success: {test_response.choices[0].message.content}"
    except Exception as e:
        results["chat_test"] = f"failed: {str(e)}"

    return JSONResponse(content={"status": "diagnostic", "results": results})


# ========== ОСНОВНОЙ ЭНДПОИНТ ==========
@app.post("/analyze")
async def analyze(request: Request):
    logging.info("Request received")

    content_type = request.headers.get("content-type", "").lower()
    text = None
    criteria = []
    upload = None

    try:
        if "application/json" in content_type:
            data = await request.json()
            text = data.get("text", "").strip() if isinstance(data, dict) else None
            criteria = normalize_criteria(data.get("criteria"))
        else:
            form = await request.form()
            text = form.get("text", "").strip() if form.get("text") else None
            criteria = normalize_criteria(form.get("criteria"))
            upload = form.get("file")
    except Exception as e:
        logging.exception("Parse error")
        raise HTTPException(status_code=400, detail="Invalid request format")

    if not text and not upload:
        raise HTTPException(status_code=400, detail="Provide text or audio file")

    dialogue_text = ""

    if upload:
        filename = getattr(upload, "filename", "audio")
        logging.info(f"Processing audio: {filename}")

        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "input_audio")
            wav_path = os.path.join(tmpdir, "audio.wav")

            try:
                content = await upload.read()
                logging.info(f"File size: {len(content)} bytes")

                with open(src_path, "wb") as f:
                    f.write(content)

                ffmpeg_to_wav(src_path, wav_path)
                logging.info("Audio converted to WAV")

                raw_text = transcribe_audio(_openai_client, wav_path)
                dialogue_text = diarize_text(_openai_client, raw_text)

            except subprocess.CalledProcessError as e:
                logging.exception("FFmpeg error")
                raise HTTPException(status_code=400, detail="Unsupported audio format")
            except Exception as e:
                logging.exception("Audio pipeline error")
                raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")
    else:
        dialogue_text = text or ""
        logging.info("Text mode")

    try:
        analysis = analyze_dialogue(_openai_client, dialogue_text, criteria)
        logging.info("Analysis completed")
    except Exception as e:
        logging.exception("Analysis error")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    return JSONResponse(content={"status": "ok", "analysis": analysis})


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)