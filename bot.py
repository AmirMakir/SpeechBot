import os
import subprocess
import requests
import librosa
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes, CommandHandler, CallbackQueryHandler
from transformers import pipeline
from dotenv import load_dotenv
import logging
import re
import nltk
from nltk.tokenize import sent_tokenize
from datetime import datetime

nltk.download('punkt_tab')

# Logging setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# -----------------------------
# Constants
# -----------------------------
FILLERS_RU = [
    "ну", "типа", "короче", "в общем", "как бы", "значит", "понимаешь", "вроде", "собственно", "это самое",
    "вообще", "ещё", "просто", "например", "я думаю", "знаешь", "ладно", "вот", "так сказать", "сразу",
    "кажется", "так", "эх", "короче говоря", "между прочим", "по сути", "как правило", "в итоге",
    "в принципе", "честно говоря", "на самом деле", "прямо", "ну вот", "кстати", "при этом",
    "если честно", "как ни странно", "пожалуй", "типа того", "так вот", "в общем-то", "сильно", "пожалуйста",
    "эээ", "эмм", "ммм", "ах", "ой", "угу", "ээ…", "эмм…", "мм…", "ах…", "ох", "эх…", "мм-хм", "ааа"
]

FILLERS_EN = [
    "um", "uh", "like", "you know", "basically", "actually", "literally", "sort of", "kind of", "i mean",
    "right", "okay", "well", "so", "anyway", "honestly", "seriously", "obviously", "definitely", "totally",
    "really", "just", "maybe", "perhaps", "probably", "essentially", "practically", "virtually", "generally",
    "apparently", "supposedly", "presumably", "allegedly", "hmm", "err", "ah", "oh", "yeah", "yep", "nah"
]

OPTIMAL_WPM_MIN = 120
OPTIMAL_WPM_MAX = 150
user_languages = {}
user_stats = {}

TRANSLATIONS = {
    'ru': {
        'welcome': """
🎤 <b>Добро пожаловать в Speech Analyzer Bot!</b>

Я помогу вам улучшить навыки публичных выступлений и речи.

<b>Что я умею:</b>
✅ Анализировать темп и ритм речи
✅ Находить слова-паразиты
✅ Оценивать выразительность и интонацию
✅ Давать персональные рекомендации
✅ Отслеживать ваш прогресс

<b>Выберите язык интерфейса:</b>
""",
        'language_selected': '✅ <b>Язык интерфейса: Русский</b>\n\nИспользуйте меню для навигации 👇',
        'main_menu': '📋 <b>Главное меню</b>\n\nВыберите действие:',
        'help_text': """
ℹ️ <b>Справка</b>

<b>Как использовать бота:</b>

1️⃣ Запишите голосовое сообщение или загрузите аудио
2️⃣ Отправьте мне его
3️⃣ Получите детальный анализ с рекомендациями

<b>Доступные команды:</b>
/start - Начать работу с ботом
/help - Показать эту справку
/stats - Посмотреть вашу статистику
/tips - Получить советы по ораторскому искусству
/settings - Настройки бота
/about - О боте и разработчике

<b>Кнопки меню:</b>
🎙 Отправить аудио - Записать голосовое сообщение
📊 Моя статистика - Ваш прогресс
💡 Советы - Полезные рекомендации
⚙️ Настройки - Изменить язык и параметры

<b>Форматы аудио:</b>
• Голосовые сообщения Telegram
• Audio файлы (MP3, OGG, WAV, M4A)
• Максимальная длительность: 10 минут

<b>Языки анализа:</b>
Русский 🇷🇺 и English 🇬🇧
""",
        'stats_title': '📊 <b>Ваша статистика</b>\n\n',
        'stats_empty': 'У вас пока нет анализов. Отправьте голосовое сообщение для начала!',
        'total_analyses': '📈 Всего анализов: {}',
        'avg_wpm': '⚡️ Средний темп: {} слов/мин',
        'avg_fillers': '🎯 Среднее количество паразитов: {}',
        'last_analysis': '🕐 Последний анализ: {}',
        'tips_title': '💡 <b>Советы по ораторскому искусству</b>\n\n',
        'tips_content': """
<b>1. Контроль темпа речи</b>
• Оптимальный темп: 120-150 слов/мин
• Делайте паузы между мыслями
• Варьируйте скорость для акцентов

<b>2. Работа с паузами</b>
• Используйте паузы вместо слов-паразитов
• Пауза 2-3 секунды = время для обдумывания
• Пауза создает драматический эффект

<b>3. Избавление от слов-паразитов</b>
• Осознайте свои "любимые" паразиты
• Заменяйте их паузами
• Практикуйтесь ежедневно

<b>4. Выразительность</b>
• Меняйте высоту голоса
• Используйте эмоции в речи
• Подчеркивайте важные слова

<b>5. Практика</b>
• Записывайте себя каждый день
• Анализируйте результаты
• Отслеживайте прогресс

<i>Используйте /stats чтобы видеть ваш прогресс!</i>
""",
        'settings_title': '⚙️ <b>Настройки</b>\n\nВыберите параметр для изменения:',
        'about_text': """
ℹ️ <b>О Speech Analyzer Bot</b>

<b>Версия:</b> 2.0
<b>Модель распознавания:</b> OpenAI Whisper Medium
<b>AI Анализ:</b> Google Gemma 3 27B

<b>Технологии:</b>
• Speech Recognition (Whisper)
• Audio Processing (Librosa)
• Natural Language Processing
• Machine Learning Analysis

<b>Разработчик:</b> @AmirMakir

💝 Если бот помог вам, расскажите о нем друзьям!

<b>GitHub:</b> github.com/AmirMakir/SpeechBot
""",
        'processing': '🎧 <b>Обрабатываю аудио...</b>\n\nЭто займет несколько секунд ⏳',
        'converting': '🔄 <b>Конвертирую аудио...</b>',
        'recognizing': '🎙 <b>Распознаю речь...</b>\n\nИспользуется AI технология Whisper 🤖',
        'analyzing': '📊 <b>Анализирую речь...</b>\n\nПроверяю темп, паузы и выразительность 🔍',
        'generating': '🤖 <b>Генерирую персональные рекомендации...</b>\n\nПочти готово! ✨',
        'error': '❌ <b>Произошла ошибка:</b>\n\n{}',
        'timeout_error': '❌ <b>Ошибка:</b> Превышено время обработки аудио\n\nПопробуйте отправить более короткий файл.',
        'analysis_title': '🎤 <b>РЕЗУЛЬТАТЫ АНАЛИЗА</b>',
        'basic_metrics': '\n📊 <b>Базовые метрики:</b>',
        'speech_quality': '\n🗣 <b>Качество речи:</b>',
        'transcription': '\n📄 <b>Полная транскрипция:</b>',
        'recommendations': '\n\n💡 <b>РЕКОМЕНДАЦИИ ДЛЯ УЛУЧШЕНИЯ:</b>\n',
        'btn_send_audio': '🎙 Отправить аудио',
        'btn_stats': '📊 Моя статистика',
        'btn_tips': '💡 Советы',
        'btn_settings': '⚙️ Настройки',
        'btn_help': '❓ Помощь',
        'btn_change_lang': '🌍 Изменить язык',
        'btn_back': '◀️ Назад',
        'analysis_complete': '✅ <b>Анализ завершен!</b>\n\nВаш анализ #{}\n\nПродолжайте практиковаться для достижения лучших результатов! 💪',
    },
    'en': {
        'welcome': """
🎤 <b>Welcome to Speech Analyzer Bot!</b>

I will help you improve your public speaking and speech skills.

<b>What I can do:</b>
✅ Analyze speech tempo and rhythm
✅ Find filler words
✅ Evaluate expressiveness and intonation
✅ Give personalized recommendations
✅ Track your progress

<b>Choose your interface language:</b>
""",
        'language_selected': '✅ <b>Interface language: English</b>\n\nUse the menu for navigation 👇',
        'main_menu': '📋 <b>Main Menu</b>\n\nChoose an action:',
        'help_text': """
ℹ️ <b>Help</b>

<b>How to use the bot:</b>

1️⃣ Record a voice message or upload audio
2️⃣ Send it to me
3️⃣ Get detailed analysis with recommendations

<b>Available commands:</b>
/start - Start working with the bot
/help - Show this help
/stats - View your statistics
/tips - Get public speaking tips
/settings - Bot settings
/about - About the bot and developer

<b>Menu buttons:</b>
🎙 Send Audio - Record voice message
📊 My Stats - Your progress
💡 Tips - Useful recommendations
⚙️ Settings - Change language and parameters

<b>Audio formats:</b>
• Telegram voice messages
• Audio files (MP3, OGG, WAV, M4A)
• Maximum duration: 10 minutes

<b>Analysis languages:</b>
Russian 🇷🇺 and English 🇬🇧
""",
        'stats_title': '📊 <b>Your Statistics</b>\n\n',
        'stats_empty': 'You have no analyses yet. Send a voice message to get started!',
        'total_analyses': '📈 Total analyses: {}',
        'avg_wpm': '⚡️ Average tempo: {} words/min',
        'avg_fillers': '🎯 Average filler words: {}',
        'last_analysis': '🕐 Last analysis: {}',
        'tips_title': '💡 <b>Public Speaking Tips</b>\n\n',
        'tips_content': """
<b>1. Speech Tempo Control</b>
• Optimal tempo: 120-150 words/min
• Make pauses between thoughts
• Vary speed for emphasis

<b>2. Working with Pauses</b>
• Use pauses instead of filler words
• 2-3 second pause = thinking time
• Pause creates dramatic effect

<b>3. Eliminating Filler Words</b>
• Recognize your "favorite" fillers
• Replace them with pauses
• Practice daily

<b>4. Expressiveness</b>
• Change voice pitch
• Use emotions in speech
• Emphasize important words

<b>5. Practice</b>
• Record yourself every day
• Analyze results
• Track progress

<i>Use /stats to see your progress!</i>
""",
        'settings_title': '⚙️ <b>Settings</b>\n\nChoose a parameter to change:',
        'about_text': """
ℹ️ <b>About Speech Analyzer Bot</b>

<b>Version:</b> 2.0
<b>Recognition Model:</b> OpenAI Whisper Medium
<b>AI Analysis:</b> Google Gemma 3 27B

<b>Technologies:</b>
• Speech Recognition (Whisper)
• Audio Processing (Librosa)
• Natural Language Processing
• Machine Learning Analysis

<b>Developer:</b> @AmirMakir

💝 If the bot helped you, tell your friends!

<b>GitHub:</b> github.com/AmirMakir/SpeechBot
""",
        'processing': '🎧 <b>Processing audio...</b>\n\nThis will take a few seconds ⏳',
        'converting': '🔄 <b>Converting audio...</b>',
        'recognizing': '🎙 <b>Recognizing speech...</b>\n\nUsing Whisper AI technology 🤖',
        'analyzing': '📊 <b>Analyzing speech...</b>\n\nChecking tempo, pauses and expressiveness 🔍',
        'generating': '🤖 <b>Generating personalized recommendations...</b>\n\nAlmost ready! ✨',
        'error': '❌ <b>An error occurred:</b>\n\n{}',
        'timeout_error': '❌ <b>Error:</b> Audio processing timeout exceeded\n\nTry sending a shorter file.',
        'analysis_title': '🎤 <b>ANALYSIS RESULTS</b>',
        'basic_metrics': '\n📊 <b>Basic metrics:</b>',
        'speech_quality': '\n🗣 <b>Speech quality:</b>',
        'transcription': '\n📄 <b>Full transcription:</b>',
        'recommendations': '\n\n💡 <b>RECOMMENDATIONS FOR IMPROVEMENT:</b>\n',
        'btn_send_audio': '🎙 Send Audio',
        'btn_stats': '📊 My Stats',
        'btn_tips': '💡 Tips',
        'btn_settings': '⚙️ Settings',
        'btn_help': '❓ Help',
        'btn_change_lang': '🌍 Change Language',
        'btn_back': '◀️ Back',
        'analysis_complete': '✅ <b>Analysis complete!</b>\n\nYour analysis #{}\n\nKeep practicing for better results! 💪',
    }
}

# -----------------------------
# Model setup
# -----------------------------
try:

    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-medium",
        chunk_length_s=30,
        return_timestamps=True,
    )

    logger.info("Model loaded successfully")

except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise

# -----------------------------
# API configuration
# -----------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BOT_TOKEN = os.getenv("BOT_TOKEN")

if not OPENROUTER_API_KEY or not BOT_TOKEN:
    raise ValueError("Environment variables OPENROUTER_API_KEY or BOT_TOKEN not set")


def get_main_keyboard(lang='en'):
    """Create main menu keyboard based on user language"""
    t = TRANSLATIONS[lang]
    keyboard = [
        [KeyboardButton(t['btn_send_audio'])],
        [KeyboardButton(t['btn_stats']), KeyboardButton(t['btn_tips'])],
        [KeyboardButton(t['btn_settings']), KeyboardButton(t['btn_help'])]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)


def query_llm(prompt, speech_lang):
    """Query Gemma 3 27B via OpenRouter API for speech analysis"""
    system_content = (
        "You are an expert in public speaking and oratory skills. You analyze speech and provide specific recommendations."
        if speech_lang == "en"
        else "Ты эксперт по ораторскому мастерству и публичным выступлениям. Анализируешь речь и даешь конкретные рекомендации."
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/AmirMakir/SpeechBot",
        "X-Title": "Speech Analyzer Bot"
    }
    payload = {
        "model": "google/gemma-3-27b-it:free",
        "messages": [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1500
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        logger.error(f"API request error: {e}")
        if hasattr(e.response, 'text'):
            logger.error(f"API response: {e.response.text}")
        return "Error getting recommendations. Please try again later." if speech_lang == "en" else "Ошибка при получении рекомендаций. Попробуйте позже."


def count_fillers(text, lang="ru"):
    """Count filler words by tokens"""
    fillers = set(FILLERS_RU if lang == "ru" else FILLERS_EN)
    words = re.findall(r"\b\w+\b", text.lower())

    filler_details = {}
    for w in words:
        if w in fillers:
            filler_details[w] = filler_details.get(w, 0) + 1

    total_count = sum(filler_details.values())
    return total_count, filler_details


def analyze_prosody(y, sr, lang="ru"):
    """Advanced expressiveness analysis: pitch, variability, energy"""
    # Pitch analysis
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=75, fmax=350)
    pitch = []
    for i in range(pitches.shape[1]):
        idx = magnitudes[:, i].argmax()
        val = pitches[idx, i]
        if val > 0:
            pitch.append(val)

    pitch = np.array(pitch)
    if len(pitch) > 10:
        pitch = librosa.effects.harmonic(pitch)

    pitch_var = float(np.std(pitch)) if len(pitch) > 1 else 0
    pitch_mean = float(np.mean(pitch)) if len(pitch) > 1 else 0

    # Energy analysis
    energy = librosa.feature.rms(y=y)[0]
    energy_var = float(np.std(energy))
    energy_mean = float(np.mean(energy))

    # Determine monotony level
    if lang == "en":
        monotony = (
            "very low (lively sound)" if pitch_var > 60 else
            "moderate" if pitch_var > 30 else
            "high (monotonous)"
        )
        dynamics = (
            "pronounced dynamics" if energy_var > 0.03 else
            "medium dynamics" if energy_var > 0.015 else
            "flat (almost no volume changes)"
        )
    else:
        monotony = (
            "очень низкая (живое звучание)" if pitch_var > 60 else
            "умеренная" if pitch_var > 30 else
            "высокая (монотонно)"
        )
        dynamics = (
            "ярко выраженная динамика" if energy_var > 0.03 else
            "средняя динамика" if energy_var > 0.015 else
            "плоская (почти нет изменений громкости)"
        )

    return {
        "pitch_mean": round(pitch_mean, 1),
        "pitch_variance": round(pitch_var, 1),
        "monotony": monotony,
        "energy_variance": round(energy_var, 4),
        "energy_mean": round(energy_mean, 4),
        "energy_rating": dynamics,
    }


def analyze_speech(audio_path, text, lang="ru"):
    """Improved speech analysis: tempo, pauses, fillers, prosody"""
    y, sr = librosa.load(audio_path, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)

    # Speed (WPM)
    words = re.findall(r"\b\w+\b", text)
    wpm = len(words) / (duration / 60) if duration > 0 else 0

    # Tempo rating based on language
    if lang == "en":
        tempo_rating = (
            "optimal" if OPTIMAL_WPM_MIN <= wpm <= OPTIMAL_WPM_MAX
            else "too slow" if wpm < OPTIMAL_WPM_MIN
            else "too fast"
        )
    else:
        tempo_rating = (
            "оптимальный" if OPTIMAL_WPM_MIN <= wpm <= OPTIMAL_WPM_MAX
            else "слишком медленный" if wpm < OPTIMAL_WPM_MIN
            else "слишком быстрый"
        )

    # Pauses detection
    pauses = detect_pauses(y, sr)
    short_pauses = sum(1 for p in pauses if 0.3 <= p <= 1.0)
    long_pauses = sum(1 for p in pauses if p > 1.5)

    # Filler words
    fillers_count, filler_details = count_fillers(text, lang)

    # Prosody analysis
    prosody = analyze_prosody(y, sr, lang)

    return {
        "duration_sec": round(duration, 2),
        "word_count": len(words),
        "words_per_minute": round(wpm, 1),
        "tempo_rating": tempo_rating,
        "short_pauses": short_pauses,
        "long_pauses": long_pauses,
        "fillers_count": fillers_count,
        "filler_details": filler_details,
        "prosody": prosody
    }


def detect_pauses(y, sr):
    """More accurate pause detection based on energy"""
    hop = 512
    frame_duration = hop / sr
    energy = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]

    threshold = np.percentile(energy, 15)  # Dynamic threshold
    silence_frames = np.where(energy < threshold)[0]

    pauses = []
    if len(silence_frames) == 0:
        return pauses

    start = silence_frames[0]
    prev = silence_frames[0]

    for f in silence_frames[1:]:
        if f - prev > 1:
            duration = (prev - start) * frame_duration
            if duration > 0.25:
                pauses.append(duration)
            start = f
        prev = f

    # Add last pause
    duration = (prev - start) * frame_duration
    if duration > 0.25:
        pauses.append(duration)

    return pauses


def analyze_text_quality(text, lang="ru"):
    """Analyze logic, structure and clarity of text"""
    language = "russian" if lang == "ru" else "english"
    sentences = sent_tokenize(text, language=language)

    avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0

    too_long = [s for s in sentences if len(s.split()) > 20]
    repetitions = {}

    words = re.findall(r"\b\w+\b", text.lower())
    for w in words:
        if len(w) > 4:
            repetitions[w] = repetitions.get(w, 0) + 1
    repetitions = {w: c for w, c in repetitions.items() if c >= 3}

    return {
        "sentence_count": len(sentences),
        "avg_sentence_length": avg_sentence_len,
        "long_sentences": too_long,
        "repetitions": repetitions
    }


def prepare_llm_prompt(text, analysis, lang="ru"):
    """Prepare prompt for LLM based on speech language"""
    filler_list = "\n".join([f"  - '{word}': {count} {'times' if lang == 'en' else 'раз'}"
                             for word, count in analysis['filler_details'].items()])

    text_quality = analyze_text_quality(text, lang)

    if lang == "en":
        prompt = f"""
Analyze the speech in English and provide specific recommendations for improvement.

📝 SPEECH TEXT:
{text}

📊 METRICS:
- Duration: {analysis['duration_sec']} sec
- Word count: {analysis['word_count']}
- Speech tempo: {analysis['words_per_minute']} words/min ({analysis['tempo_rating']}, norm: {OPTIMAL_WPM_MIN}-{OPTIMAL_WPM_MAX})
- Short pauses: {analysis['short_pauses']}
- Long pauses (hesitations): {analysis['long_pauses']}
- Filler words: {analysis['fillers_count']} times
{filler_list if filler_list else "  (none detected)"}
- Monotony: {analysis['prosody']['monotony']}
- Volume dynamics: {analysis['prosody']['energy_rating']}

🧠 TEXT STRUCTURE ANALYSIS:
- Sentence count: {text_quality['sentence_count']}
- Average sentence length: {text_quality['avg_sentence_length']:.1f} words
- Too long sentences: {len(text_quality['long_sentences'])}
- Frequent word repetitions: {text_quality['repetitions']}

📋 TASK
1) Rate the speech on a scale of 1–10
Evaluate the following parameters:
 - Correctness (pronunciation + grammar)
 - Logic (structure and coherence)
 - Clarity (clear expression)
 - Speech purity (absence of fillers)
 - Expressiveness (intonation, pause work, emotions)

2) You can only use these HTML tags: <b>, <i>, <u>, <code>, <pre>, <a>, <blockquote>. (other tags are strictly prohibited):

3) Give 3–5 recommendations for speech improvement
Recommendations should be:
 - specific,
 - implementable,
 - based on metrics and text.

Focus on:
 - tempo,
 - diction,
 - fillers,
 - structure,
 - expressiveness.

4) Find problematic places in the text

Show specific fragments that sound weak, and suggest 2–3 improved versions of each.

Response format (strictly):
 - 5 ratings (1–10)
 - 3–5 recommendations in list form
 - Reformulations of problematic phrases
Write concisely, structured and to the point.
"""
    else:
        prompt = f"""
Проанализируй речь на русском языке и дай конкретные рекомендации для улучшения.

📝 ТЕКСТ РЕЧИ:
{text}

📊 МЕТРИКИ:
- Длительность: {analysis['duration_sec']} сек
- Количество слов: {analysis['word_count']}
- Темп речи: {analysis['words_per_minute']} слов/мин ({analysis['tempo_rating']}, норма: {OPTIMAL_WPM_MIN}-{OPTIMAL_WPM_MAX})
- Короткие паузы: {analysis['short_pauses']}
- Длинные паузы (заминки): {analysis['long_pauses']}
- Слова-паразиты: {analysis['fillers_count']} раз
{filler_list if filler_list else "  (не обнаружено)"}
- Монотонность: {analysis['prosody']['monotony']}
- Динамика громкости: {analysis['prosody']['energy_rating']}

🧠 АНАЛИЗ СТРУКТУРЫ ТЕКСТА:
- Количество предложений: {text_quality['sentence_count']}
- Средняя длина предложения: {text_quality['avg_sentence_length']:.1f} слов
- Слишком длинные предложения: {len(text_quality['long_sentences'])}
- Частые повторы слов: {text_quality['repetitions']}

📋 ЗАДАЧА
1) Оцени речь по шкале 1–10
Оцени следующие параметры:
 - Правильность (произношение + грамматика)
 - Логичность (структура и связность)
 - Понятность (ясность изложения)
 - Чистота речи (отсутствие паразитов)
 - Выразительность (интонация, работа с паузами, эмоции)

2) ты можешь использовать только эти HTML теги: <b>, <i>, <u>, <code>, <pre>, <a>, <blockquote>. (другие тебе строго запрещено использовать) :

3) Дай 3–5 рекомендаций по улучшению речи
Рекомендации должны быть:
 - конкретными,
 - реализуемыми,
 - основанными на метриках и тексте.

Сфокусируйся на:
 - темпе,
 - дикции,
 - паразитах,
 - структуре,
 - выразительности.

4) Найди проблемные места в тексте

Покажи конкретные фрагменты, которые звучат слабо, и предложи 2–3 улучшенных варианта каждого.

Формат ответа (строго):
 - 5 оценок (1–10)
 - 3–5 рекомендаций списком
 - Переформулировки проблемных фраз
Пиши кратко, структурировано и по делу.
"""
    return prompt


def format_analysis_response(text, analysis, recommendations, model_name, ui_lang="ru"):
    """Format analysis response based on UI language"""
    t = TRANSLATIONS[ui_lang]

    response = f"{t['analysis_title']}\n"
    response += f"━━━━━━━━━━━━━━━━━━━━\n\n"

    response += f"{t['basic_metrics']}\n"
    response += f"⏱ {analysis['duration_sec']} sec | "
    response += f"📝 {analysis['word_count']} words\n"
    response += f"⚡️ {analysis['words_per_minute']} wpm ({analysis['tempo_rating']})\n"
    response += f"⏸ Pauses: {analysis['short_pauses']} short, {analysis['long_pauses']} long\n"
    response += f"🎯 Fillers: {analysis['fillers_count']}\n"

    response += f"{t['speech_quality']}\n"
    response += f"🎵 {analysis['prosody']['monotony']}\n"
    response += f"🔊 {analysis['prosody']['energy_rating']}\n"

    response += f"{t['transcription']}\n"
    response += f"<code>{text[:500]}{'...' if len(text) > 500 else ''}</code>\n"

    response += f"{t['recommendations']}"
    response += f"{recommendations}"

    return response


def detect_language(text):
    """Simple language detection based on character patterns"""
    # Count Cyrillic vs Latin characters
    cyrillic_chars = len(re.findall(r'[а-яА-ЯёЁ]', text))
    latin_chars = len(re.findall(r'[a-zA-Z]', text))

    # If more than 60% Cyrillic, it's Russian
    if cyrillic_chars > latin_chars and cyrillic_chars > len(text) * 0.3:
        return "ru"
    return "en"


def update_user_stats(user_id, analysis):
    """Update user statistics after analysis"""
    if user_id not in user_stats:
        user_stats[user_id] = {
            'total_analyses': 0,
            'total_wpm': 0,
            'total_fillers': 0,
            'last_analysis_date': None,
            'analyses_history': []
        }

    stats = user_stats[user_id]
    stats['total_analyses'] += 1
    stats['total_wpm'] += analysis['words_per_minute']
    stats['total_fillers'] += analysis['fillers_count']
    stats['last_analysis_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')

    # Store last 10 analyses
    stats['analyses_history'].append({
        'date': stats['last_analysis_date'],
        'wpm': analysis['words_per_minute'],
        'fillers': analysis['fillers_count'],
        'duration': analysis['duration_sec']
    })
    if len(stats['analyses_history']) > 10:
        stats['analyses_history'].pop(0)


# -----------------------------
# Command Handlers
# -----------------------------

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command with language selection"""
    keyboard = [
        [
            InlineKeyboardButton("🇷🇺 Русский", callback_data='lang_ru'),
            InlineKeyboardButton("🇬🇧 English", callback_data='lang_en')
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    user_id = update.effective_user.id
    welcome_lang = user_languages.get(user_id, 'en')

    await update.message.reply_text(
        TRANSLATIONS[welcome_lang]['welcome'],
        reply_markup=reply_markup,
        parse_mode='HTML'
    )


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    user_id = update.effective_user.id
    lang = user_languages.get(user_id, 'en')
    t = TRANSLATIONS[lang]

    # Send help text and ensure it's displayed
    await update.message.reply_text(
        t['help_text'],
        parse_mode='HTML',
        reply_markup=get_main_keyboard(lang),
        disable_web_page_preview=True
    )


async def stats_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /stats command - show user statistics"""
    user_id = update.effective_user.id
    lang = user_languages.get(user_id, 'en')
    t = TRANSLATIONS[lang]

    if user_id not in user_stats or user_stats[user_id]['total_analyses'] == 0:
        await update.message.reply_text(
            t['stats_empty'],
            parse_mode='HTML',
            reply_markup=get_main_keyboard(lang)
        )
        return

    stats = user_stats[user_id]
    avg_wpm = round(stats['total_wpm'] / stats['total_analyses'], 1)
    avg_fillers = round(stats['total_fillers'] / stats['total_analyses'], 1)

    response = t['stats_title']
    response += f"{t['total_analyses'].format(stats['total_analyses'])}\n"
    response += f"{t['avg_wpm'].format(avg_wpm)}\n"
    response += f"{t['avg_fillers'].format(avg_fillers)}\n"
    response += f"{t['last_analysis'].format(stats['last_analysis_date'])}\n\n"

    # Add progress chart
    response += "<b>📈 Recent Progress:</b>\n"
    for i, analysis in enumerate(stats['analyses_history'][-5:], 1):
        response += f"{i}. {analysis['date']}: {analysis['wpm']} wpm, {analysis['fillers']} fillers\n"

    await update.message.reply_text(
        response,
        parse_mode='HTML',
        reply_markup=get_main_keyboard(lang)
    )


async def tips_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /tips command"""
    user_id = update.effective_user.id
    lang = user_languages.get(user_id, 'en')
    t = TRANSLATIONS[lang]

    # Send tips with proper formatting
    message = t['tips_title'] + t['tips_content']
    await update.message.reply_text(
        message,
        parse_mode='HTML',
        reply_markup=get_main_keyboard(lang),
        disable_web_page_preview=True
    )


async def settings_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /settings command"""
    user_id = update.effective_user.id
    lang = user_languages.get(user_id, 'en')
    t = TRANSLATIONS[lang]

    keyboard = [
        [InlineKeyboardButton(t['btn_change_lang'], callback_data='change_lang')],
        [InlineKeyboardButton(t['btn_back'], callback_data='back_to_menu')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        t['settings_title'],
        reply_markup=reply_markup,
        parse_mode='HTML'
    )


async def about_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /about command"""
    user_id = update.effective_user.id
    lang = user_languages.get(user_id, 'en')
    t = TRANSLATIONS[lang]

    await update.message.reply_text(
        t['about_text'],
        parse_mode='HTML',
        reply_markup=get_main_keyboard(lang),
        disable_web_page_preview=True
    )


async def text_menu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text button presses from menu"""
    user_id = update.effective_user.id
    lang = user_languages.get(user_id, 'en')
    t = TRANSLATIONS[lang]
    text = update.message.text

    if text == t['btn_help'] or text == '❓ Help' or text == '❓ Помощь':
        await help_handler(update, context)
    elif text == t['btn_stats'] or text == '📊 My Stats' or text == '📊 Моя статистика':
        await stats_handler(update, context)
    elif text == t['btn_tips'] or text == '💡 Tips' or text == '💡 Советы':
        await tips_handler(update, context)
    elif text == t['btn_settings'] or text == '⚙️ Settings' or text == '⚙️ Настройки':
        await settings_handler(update, context)
    elif text == t['btn_send_audio'] or text == '🎙 Send Audio' or text == '🎙 Отправить аудио':
        prompt_text = "🎤 <b>Ready to analyze!</b>\n\nPlease send a voice message or audio file." if lang == 'en' else "🎤 <b>Готов к анализу!</b>\n\nОтправьте голосовое сообщение или аудиофайл."
        await update.message.reply_text(prompt_text, parse_mode='HTML')


async def language_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle language selection and change"""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id

    if query.data == 'change_lang':
        keyboard = [
            [
                InlineKeyboardButton("🇷🇺 Русский", callback_data='lang_ru'),
                InlineKeyboardButton("🇬🇧 English", callback_data='lang_en')
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "🌍 <b>Choose language / Выберите язык:</b>",
            reply_markup=reply_markup,
            parse_mode='HTML'
        )
    elif query.data.startswith('lang_'):
        selected_lang = query.data.split('_')[1]
        user_languages[user_id] = selected_lang
        t = TRANSLATIONS[selected_lang]

        # Edit message to confirm language selection
        await query.edit_message_text(
            t['language_selected'],
            parse_mode='HTML'
        )

        # Send detailed welcome message with instructions
        welcome_detail = (
            "🎯 <b>Как начать:</b>\n\n"
            "1️⃣ Нажмите кнопку <b>🎙 Отправить аудио</b> внизу\n"
            "2️⃣ Или просто запишите голосовое сообщение\n"
            "3️⃣ Получите детальный анализ вашей речи\n\n"
            "💡 <b>Совет:</b> Говорите естественно, как обычно. "
            "Я проанализирую темп, паузы, слова-паразиты и дам рекомендации!\n\n"
            "📱 Используйте меню внизу для навигации или команды:\n"
            "/help - Подробная справка\n"
            "/stats - Ваша статистика\n"
            "/tips - Советы по ораторству"
            if selected_lang == 'ru' else
            "🎯 <b>How to start:</b>\n\n"
            "1️⃣ Press the <b>🎙 Send Audio</b> button below\n"
            "2️⃣ Or just record a voice message\n"
            "3️⃣ Get detailed analysis of your speech\n\n"
            "💡 <b>Tip:</b> Speak naturally, as you normally do. "
            "I'll analyze tempo, pauses, filler words and give recommendations!\n\n"
            "📱 Use the menu below for navigation or commands:\n"
            "/help - Detailed help\n"
            "/stats - Your statistics\n"
            "/tips - Speaking tips"
        )

        await context.bot.send_message(
            chat_id=user_id,
            text=welcome_detail,
            reply_markup=get_main_keyboard(selected_lang),
            parse_mode='HTML'
        )

        logger.info(f"User {user_id} selected language: {selected_lang}")
    elif query.data == 'back_to_menu':
        lang = user_languages.get(user_id, 'en')
        t = TRANSLATIONS[lang]
        await query.edit_message_text(t['main_menu'], parse_mode='HTML')


async def audio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Audio message handler (supports Russian and English)"""
    user_id = update.effective_user.id
    ui_lang = user_languages.get(user_id, 'en')
    t = TRANSLATIONS[ui_lang]

    logger.info(f"Received audio from user {user_id}")

    input_path = None
    wav_path = None

    try:
        # Send processing status message
        status_message = await update.message.reply_text(t['processing'], parse_mode='HTML')

        # 1. Download audio
        audio_file = await update.message.audio.get_file() if update.message.audio else await update.message.voice.get_file()
        input_path = f"temp_audio_{user_id}.ogg"
        await audio_file.download_to_drive(input_path)
        logger.info(f"Audio saved: {input_path}")

        # 2. Convert to WAV
        await status_message.edit_text(t['converting'], parse_mode="HTML")
        wav_path = f"temp_audio_{user_id}.wav"
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=60
        )
        if result.returncode != 0:
            raise Exception("Audio conversion error")
        logger.info(f"Audio converted: {wav_path}")

        # 3. Transcription
        await status_message.edit_text(t['recognizing'], parse_mode="HTML")
        result = pipe(wav_path)

        text = result["text"]
        chunks = result.get("chunks", [])
        model_name = "Whisper Medium"

        # Detect speech language
        speech_lang = detect_language(text)
        logger.info(f"Detected speech language: {speech_lang}")

        logger.info(f"Transcription complete. Text: {len(text)} chars, segments: {len(chunks)}")

        # 4. Speech analysis
        await status_message.edit_text(t['analyzing'], parse_mode="HTML")
        analysis = analyze_speech(wav_path, text, speech_lang)

        # 5. Get LLM recommendations
        await status_message.edit_text(t['generating'], parse_mode="HTML")
        prompt = prepare_llm_prompt(text, analysis, speech_lang)
        recommendations = query_llm(prompt, speech_lang)

        # 6. Update user statistics
        update_user_stats(user_id, analysis)

        # 7. Format and send response
        response = format_analysis_response(text, analysis, recommendations, model_name, ui_lang)

        # Delete status message
        await status_message.delete()

        # Send result (split if too long)
        if len(response) > 4096:
            parts = [response[i:i + 4096] for i in range(0, len(response), 4096)]
            for part in parts:
                await update.message.reply_text(part, parse_mode="HTML")
        else:
            await update.message.reply_text(response, parse_mode="HTML")

        # Send completion message with stats
        total_analyses = user_stats[user_id]['total_analyses']
        completion_msg = t['analysis_complete'].format(total_analyses)
        await update.message.reply_text(
            completion_msg,
            parse_mode='HTML',
            reply_markup=get_main_keyboard(ui_lang)
        )

        logger.info(f"Analysis complete for user {user_id}")

    except subprocess.TimeoutExpired:
        await update.message.reply_text(t['timeout_error'], parse_mode='HTML')
        logger.error("Timeout during audio conversion")
    except Exception as e:
        error_msg = t['error'].format(str(e))
        await update.message.reply_text(error_msg, parse_mode='HTML')
        logger.error(f"Processing error: {e}", exc_info=True)
    finally:
        # Clean up temporary files
        for path in [input_path, wav_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info(f"Deleted temporary file: {path}")
                except Exception as e:
                    logger.error(f"Could not delete {path}: {e}")


# -----------------------------
# Bot startup
# -----------------------------
def main():
    """Main function to start the bot"""
    logger.info("Starting bot...")

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Add command handlers
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(CommandHandler("stats", stats_handler))
    app.add_handler(CommandHandler("tips", tips_handler))
    app.add_handler(CommandHandler("settings", settings_handler))
    app.add_handler(CommandHandler("about", about_handler))

    # Add callback handlers
    app.add_handler(CallbackQueryHandler(language_callback))

    # Add message handlers
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_menu_handler))
    app.add_handler(MessageHandler(filters.AUDIO | filters.VOICE, audio_handler))

    logger.info("✅ Bot started and ready to work!")
    logger.info("Available commands: /start, /help, /stats, /tips, /settings, /about")
    app.run_polling()


if __name__ == "__main__":
    main()