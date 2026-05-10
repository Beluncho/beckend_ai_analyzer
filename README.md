# 🎙️ AI Call Analyzer

Анализ аудиозаписей звонков и текстовых диалогов с помощью LLM.

## Возможности

- 🎧 Транскрибация аудио (mp3, wav, m4a, ogg)
- 🗣️ Автоматическая диаризация (разделение по спикерам)
- 📊 Анализ диалога по любым критериям
- 📝 Поддержка текстового ввода (если расшифровка уже есть)

## 🚀 Деплой Backend:
* Build Command:
```
pip install -r requirements.txt
```

* Start Command (Render for example): 
```
uvicorn app:app --host 0.0.0.0 --port $PORT
```

* Local Start: 
```
uvicorn app:app --reload --port 8000
```
Сервер будет доступен по адресу: http://localhost:8000

## 📁 Структура проекта
```text
.
├── app.py                 # основной код сервера
├── requirements.txt       # зависимости
├── .env                   # переменные окружения (не в репозитории)
├── .gitignore             # игнорируемые файлы
└── README.md              # документация
```
## 🔐 Переменные окружения
```text
PROXY_API_KEY=sk-ваш_ключ_от_proxyapi.ru
Для деплоя на Render добавь эту же переменную в **Environment Variables**
```
## 🛠️ Технологии
```text
FastAPI — веб-фреймворк
OpenAI (через прокси) — транскрибация, диаризация, анализ
FFmpeg — конвертация аудио
Render — хостинг
```

## 📄 Лицензия
MIT