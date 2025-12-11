# 🎤 Speech Analyzer Bot

A Telegram bot that analyzes speech quality, provides recommendations, and helps improve public speaking skills.

## ✨ Features

- 🎙️ **Speech Recognition**: Automatic transcription using OpenAI Whisper
- 📊 **Speech Analysis**: Tempo, pauses, filler words detection
- 🎵 **Prosody Analysis**: Pitch, intonation, and dynamics evaluation
- 🤖 **AI Recommendations**: Personalized tips from Gemma 3 27B
- 🌍 **Bilingual Support**: English and Russian interface
- 🔍 **Automatic Language Detection**: Analyzes speech in detected language

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- FFmpeg installed on your system
- Telegram Bot Token
- OpenRouter API Key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/AmirMakir/SpeechBot.git
cd SpeechBot
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install FFmpeg:
- **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

5. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

6. Run the bot:
```bash
python bot.py
```

## 🔑 Getting API Keys

### Telegram Bot Token
1. Open [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/newbot`
3. Follow instructions to create your bot
4. Copy the token provided

### OpenRouter API Key
1. Visit [openrouter.ai](https://openrouter.ai/)
2. Sign up for an account
3. Go to API Keys section
4. Create a new API key
5. Copy the key

## 🐳 Docker Deployment (Optional)

Build and run with Docker:
```bash
docker build -t speechbot .
docker run -d --env-file .env speechbot
```

## 📊 Usage

1. Start conversation: `/start`
2. Select your preferred language (🇷🇺/🇬🇧)
3. Send a voice message or audio file
4. Receive detailed analysis and recommendations

## 🛠️ Tech Stack

- **Python 3.9+**
- **python-telegram-bot**: Telegram Bot API
- **Transformers**: Whisper speech recognition
- **Librosa**: Audio analysis
- **OpenRouter**: Gemma 3 27B API access
- **NLTK**: Natural language processing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

Nickname (tg): [@AmirMakir](https://t.me/AmirMakir)   
Medium story:[https://medium.com/@AmirMak/...](https://medium.com/@AmirMak/i-built-an-ai-system-that-analyzes-human-speech-de60f352020c)    
Twitter: https://x.com/AmirMakirov      
Project Link: [https://github.com/AmirMakir/SpeechBot](https://github.com/AmirMakir/SpeechBot)