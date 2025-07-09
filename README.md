# ASR FINAL PROJECT: OrderBot - Ordering System in Coffee Shop utilizing Voice-based ChatBot

A voice-controlled ordering system for Highland Coffee that combines speech-to-text (STT) capabilities with an intelligent chatbot for seamless coffee ordering experience.


## Setup Instructions

### 1. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the project root directory:

```bash
touch .env
```

Add your OpenAI API key to the `.env` file:

```
OPENAI_API_KEY=your_openai_api_key_here
```

**Note:** You need a valid OpenAI API key to use the chatbot functionality. Get one from [OpenAI's website](https://platform.openai.com/api-keys).

### 4. Additional System Requirements

For audio processing, you may need to install system dependencies:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio
```

**macOS:**
```bash
brew install portaudio
```

**Note**: The Program is Running Whisper-medium model locally!

## How to Run

### Option 1: Web Interface (Recommended)

Run the main application with Gradio web interface:

```bash
python main.py
```

This will start a web server (usually at `http://localhost:7860`) where you can:
- View real-time conversation
- See system status
- Interact with the voice ordering system

### Option 2: Command Line Interface

Run the terminal-based version:

```bash
python main_no_UI.py
```

This provides a simpler command-line interface for voice ordering.


## Usage

1. **Start the application** using one of the methods above
2. **Speak your order** when prompted (e.g., "I want a Green Tea Freeze")
3. **Wait for processing** - the system will transcribe your speech and process the order
4. **Review the response** from the OrderBot
5. **Continue the conversation** to modify or complete your order

## Acknowledgments

This project utilizes several powerful APIs and models:

- **[OpenAI Whisper](https://openai.com/research/whisper)** - Advanced speech recognition model for accurate speech-to-text conversion
- **[OpenAI GPT API](https://platform.openai.com/)** - Large language model for intelligent chatbot conversations
- **[Gradio](https://gradio.app/)** - User-friendly web interface framework
- **[PyTorch](https://pytorch.org/)** - Deep learning framework for model inference
- **[Transformers by Hugging Face](https://huggingface.co/transformers/)** - Model loading and processing utilities

Special thanks to OpenAI for providing robust speech recognition and language understanding capabilities that make this voice ordering system possible.


**Note:** This project is designed for educational and demonstration purposes.
