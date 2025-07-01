#!/usr/bin/env python3
"""
Simple Speech-to-Text system using Whisper with Voice Activity Detection
No GUI - Command line interface only
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pyaudio
import numpy as np
import soundfile as sf
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoFeatureExtractor
import webrtcvad
import time


class SimpleSTT:
    """Simple Speech-to-Text class using Whisper model with VAD"""
    
    def __init__(self, model_name="openai/whisper-medium"):
        print(f"Initializing STT system with {model_name}...")
        
        # Audio parameters
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1        # Mono
        self.RATE = 16000        # 16kHz for VAD compatibility
        self.CHUNK = 320         # 20ms frames for VAD (16000 * 0.02)
        self.OUTPUT_FILENAME = "recorded_audio.wav"
        
        # VAD parameters
        self.vad_aggressiveness = 2  # 0-3, higher = more aggressive
        self.min_speech_duration = 0.5  # Minimum seconds of speech to process
        self.max_silence_duration = 2.0  # Max seconds of silence before stopping
        self.energy_threshold = 500  # Energy threshold for basic voice detection
        
        # Initialize VAD
        self.vad = webrtcvad.Vad(self.vad_aggressiveness)
        
        # Load Whisper model
        print("Loading Whisper model...")
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
        print("✅ STT system ready!")

    def calculate_energy(self, audio_frame):
        """Calculate energy of audio frame"""
        audio_data = np.frombuffer(audio_frame, dtype=np.int16)
        return np.sqrt(np.mean(audio_data**2))

    def is_speech(self, audio_frame):
        """Determine if audio frame contains speech using both VAD and energy"""
        # Energy-based detection
        energy = self.calculate_energy(audio_frame)
        energy_speech = energy > self.energy_threshold
        
        # WebRTC VAD detection
        try:
            vad_speech = self.vad.is_speech(audio_frame, self.RATE)
        except:
            vad_speech = False
        
        # Combine both methods (OR logic)
        return energy_speech or vad_speech

    def record_audio(self, max_duration=30):
        """Record audio with Voice Activity Detection"""
        audio = pyaudio.PyAudio()
        
        try:
            stream = audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            print("🎤 Listening for speech... (speak now)")
            
            # Recording state variables
            is_recording = False
            speech_frames = []
            silence_counter = 0
            speech_counter = 0
            max_frames = int(self.RATE / self.CHUNK * max_duration)
            
            # Threshold calculations
            silence_threshold_frames = int(self.max_silence_duration * self.RATE / self.CHUNK)
            speech_threshold_frames = int(self.min_speech_duration * self.RATE / self.CHUNK)
            
            for frame_count in range(max_frames):
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                has_speech = self.is_speech(data)
                
                if has_speech:
                    if not is_recording:
                        print("🗣️  Speech detected, recording...")
                        is_recording = True
                    
                    speech_frames.append(data)
                    speech_counter += 1
                    silence_counter = 0
                    
                else:  # No speech
                    if is_recording:
                        silence_counter += 1
                        speech_frames.append(data)  # Continue recording during brief silences
                        
                        # Stop if silence is too long
                        if silence_counter >= silence_threshold_frames:
                            print("⏹️  Silence detected, stopping recording...")
                            break
                    else:
                        # Reset counters when not recording
                        speech_counter = 0
                        silence_counter = 0
                
                # Show activity indicator every 10 frames
                if frame_count % 10 == 0:
                    indicator = "🔴" if is_recording else "⚪"
                    activity = "█" if has_speech else "░"
                    print(f"\r{indicator} {activity}", end="", flush=True)
            
            print()  # New line after indicator
            
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
        
        # Check if we recorded enough speech
        if speech_counter < speech_threshold_frames:
            print("❌ Not enough speech detected")
            return None
        
        # Convert frames to audio data
        if speech_frames:
            audio_data = np.frombuffer(b"".join(speech_frames), dtype=np.int16)
            return audio_data
        
        return None

    def save_audio(self, audio_data, filename=None):
        """Save audio data to file"""
        filename = filename or self.OUTPUT_FILENAME
        sf.write(filename, audio_data, self.RATE, subtype='PCM_16')
        print(f"💾 Audio saved to: {filename}")

    def transcribe_audio(self, audio_path=None):
        """Transcribe audio file to text"""
        audio_path = audio_path or self.OUTPUT_FILENAME
        print("🔄 Transcribing audio...")
        
        try:
            result = self.transcriber(audio_path)
            return result["text"]
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return None

    def run_continuous(self):
        """Run continuous speech recognition"""
        print("🚀 Starting continuous STT system")
        print("Press Ctrl+C to stop")
        print("="*50)
        
        session_count = 0
        
        try:
            while True:
                session_count += 1
                print(f"\n📢 Session #{session_count}")
                
                # Process single recording
                raw_text, corrected_text = self.process_single_recording()
                
                if raw_text:
                    print(f"\n📝 Results:")
                    print(f"   Transcription: {raw_text}")
                else:
                    print("❌ No speech detected, continuing...")
                
                # Brief pause before next session
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping STT system...")
            print("👋 Goodbye!")


def main():
    """Main function"""
    print("🎤 Simple Speech-to-Text System")
    print("===============================")
    
    try:
        # Initialize STT system
        stt = SimpleSTT(model_name="openai/whisper-medium")
        
        # Show menu
        print("\nChoose an option:")
        print("1. Continuous recording")
        print("2. Exit")
        
        while True:
            choice = input("\nEnter your choice (1-2): ").strip()
                    
            if choice == "1":
                stt.run_continuous()
                break
                
            elif choice == "2":
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1, 2")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your setup and try again.")


if __name__ == "__main__":
    main()
