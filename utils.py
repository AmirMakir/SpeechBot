import re

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
        'recognizing': '🎙 <b>Распознаю речь...</b>',
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
        'recognizing': '🎙 <b>Recognizing speech...</b>',
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


ALLOWED_TAGS = ["b", "i", "u", "code", "pre", "a", "blockquote"]

def sanitize_html(text):
    allowed = "|".join(ALLOWED_TAGS)

    text = re.sub(
        rf"</?(?!({allowed})(\s+href=\"[^\"]+\")?)[a-zA-Z0-9]+.*?>",
        "",
        text,
        flags=re.IGNORECASE
    )

    return text