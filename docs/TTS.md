# Озвучка ответов

Озвучка работает локально. Движок выбирается переменной `TTS_ENGINE` в `.env`.

| Движок | Языки | Нужно установить | Голоса |
|---|---|---|---|
| `piper` (по умолчанию) | русский, английский и любые другие, для которых есть голос Piper | `piper-tts` (без torch и FFmpeg) | по одному файлу голоса на язык |
| `silero` | русский | torch | 5 встроенных голосов |

Если озвучка не работает, интерфейс переходит на голос браузера. Чат от этого не страдает.

## Piper (рекомендуется)

**1. Установить пакет.** Одна команда, torch и FFmpeg не нужны:

    .venv\Scripts\python.exe -m pip install -r requirements-tts.txt

**2. Скачать голоса.** Голоса лежат в репозитории `rhasspy/piper-voices` на huggingface.co. Для каждого голоса нужны два файла в одной папке: `.onnx` и `.onnx.json`.
Рекомендуемые варианты:
- русский: `ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx`;
- английский: `en/en_US/lessac/medium/en_US-lessac-medium.onnx`.

Если путь в репозитории изменился, найдите голос по имени (`ru_RU-irina`) на странице репозитория. Положите файлы, например, в `C:\piper-voices\`.

**3. Указать в `.env`:**

    TTS_ENGINE=piper
    TTS_ENABLED=1
    PIPER_VOICES=ru-irina=C:\piper-voices\ru_RU-irina-medium.onnx;en-lessac=C:\piper-voices\en_US-lessac-medium.onnx

Несколько голосов задаются в одной строке через «;». Формат каждого элемента: `id=путь`. id обязан начинаться с языка: `ru-` или `en-`. Например, чтобы добавить второй русский голос:

    PIPER_VOICES=ru-irina=C:\piper-voices\ru_RU-irina-medium.onnx;ru-dmitri=C:\piper-voices\ru_RU-dmitri-medium.onnx;en-lessac=C:\piper-voices\en_US-lessac-medium.onnx

Все голоса появятся в списке выбора голоса в интерфейсе. Первый голос каждого языка используется по умолчанию для этого языка.
Английский голос необязателен: без него английский текст читается русским голосом.

**4. Перезапустить сервер** (`run_server.bat stop`, затем `run_server.bat`) и проверить `http://127.0.0.1:8000/api/tts/status`: `available` должно быть `true`, а `missing_voice_files` — пустым.

**Как выбирается голос.** Если выбран голос в интерфейсе, озвучка идёт им. Иначе язык определяется по тексту: если латиницы заметно больше кириллицы, берётся первый английский голос, иначе первый русский.

## Silero (только русский)

Нужен torch: `.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.
В `.env`: `TTS_ENGINE=silero`, `SILERO_ENABLED=1`. Модель `v4_ru.pt` скачивается с models.silero.ai в `data_unified\tts\`
либо включается `SILERO_AUTODOWNLOAD=1`. Синтез вызывается с расстановкой ударений и восстановлением «ё».

## Общее

- Перед озвучкой текст очищается: код, таблицы, ссылки и разметка не читаются вслух. `%`, `₽`, `&`, `№` читаются словами.
- Ответ делится на фразы; озвучка идёт по фразам.
- Проверка состояния: `GET /api/tts/status`, поля `engine`, `ready`, `available`, `error`, `voices`, `missing_voice_files`.
- Качество звука на вашей машине в песочнице не проверялось.
