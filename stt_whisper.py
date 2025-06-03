import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pyaudio
import numpy as np
import soundfile as sf
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoFeatureExtractor


class WhisperSTT:
    """Speech-to-Text class using Whisper model"""
    
    def __init__(self, model_name="openai/whisper-medium"):
        # Audio parameters
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1        # Mono
        self.RATE = 16000        # 16kHz
        self.CHUNK = 1024
        self.RECORD_SECONDS = 5  # Recording duration
        self.OUTPUT_FILENAME = "output.wav"
        
        # Load Whisper model components
        print(f"Loading {model_name} model...")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        
        # Initialize ASR pipeline
        self.transcriber = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.tokenizer,
            feature_extractor=self.feature_extractor,
            stride_length_s=0
        )
        print("Model loaded successfully!")

    def record_audio(self, duration=None, output_filename=None):
        """Record audio from microphone"""
        duration = duration or self.RECORD_SECONDS
        output_filename = output_filename or self.OUTPUT_FILENAME
        
        audio = pyaudio.PyAudio()
        
        try:
            stream = audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            print(f"🎤 Recording for {duration} seconds...")
            frames = []
            
            for _ in range(0, int(self.RATE / self.CHUNK * duration)):
                data = stream.read(self.CHUNK)
                frames.append(data)
            
            print("✅ Recording finished")
            
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
        
        # Convert to numpy array and save as WAV
        audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)
        sf.write(output_filename, audio_data, self.RATE, subtype='PCM_16')
        print(f"💾 Audio saved: {output_filename}")

    def transcribe_audio(self, audio_path=None):
        """Transcribe audio file to text"""
        audio_path = audio_path or self.OUTPUT_FILENAME
        
        print("🔄 Transcribing audio...")
        output = self.transcriber(audio_path)
        return output["text"]

    def run_continuous_stt(self):
        """Run continuous speech-to-text"""
        print("🚀 Starting continuous Speech-to-Text")
        print("Press Ctrl+C to stop")
        print("=" * 50)
        
        try:
            while True:
                # Record audio
                self.record_audio()
                
                # Transcribe
                text = self.transcribe_audio()
                
                # Display result
                print(f"📝 Transcription: {text}")
                print("-" * 50)
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopping program...")


def main():
    """Main function"""
    # Initialize STT system
    stt = WhisperSTT(model_name="openai/whisper-medium")
    
    # Run continuous STT
    stt.run_continuous_stt()


if __name__ == "__main__":
    main()