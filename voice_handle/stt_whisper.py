import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pyaudio
import numpy as np
import soundfile as sf
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoFeatureExtractor
from openai import OpenAI
from dotenv import load_dotenv
import os
import librosa
from scipy.signal import butter, filtfilt
import webrtcvad
import collections
import time

# Import denoising functions
from denoise_audio import (
    comprehensive_denoise, 
    spectral_subtraction, 
    bandpass_filter, 
    noise_gate, 
    adaptive_filter_denoise
)

# Load environment variables
load_dotenv()


class STT_module:
    """Speech-to-Text class using Whisper model with Voice Activity Detection - GUI Compatible"""
    
    def __init__(self, model_name="openai/whisper-medium", enable_denoising=True, status_callback=None):
        # Audio parameters
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1        # Mono
        self.RATE = 16000        # 16kHz for VAD compatibility
        self.CHUNK = 320         # 20ms frames for VAD (16000 * 0.02)
        self.OUTPUT_FILENAME = "output.wav"
        self.DENOISED_FILENAME = "output_denoised.wav"
        
        # VAD parameters
        self.vad_aggressiveness = 2  # 0-3, higher = more aggressive
        self.min_speech_duration = 0.5  # Minimum seconds of speech to process
        self.max_silence_duration = 2.0  # Max seconds of silence before stopping
        self.energy_threshold = 500  # Energy threshold for basic voice detection
        
        # Denoising settings
        self.enable_denoising = enable_denoising
        
        # Callback for status updates
        self.status_callback = status_callback
        
        # Initialize VAD
        self.vad = webrtcvad.Vad(self.vad_aggressiveness)
        
        # Initialize OpenAI client for LLM correction
        self.client = OpenAI()
        
        # Load Whisper model components
        self._update_status(f"Loading {model_name} model...")
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
        self._update_status("Model loaded successfully!")

    def _update_status(self, message):
        """Update status through callback if available"""
        if self.status_callback:
            self.status_callback(message)

    def calculate_energy(self, audio_frame):
        """Calculate energy of audio frame"""
        audio_data = np.frombuffer(audio_frame, dtype=np.int16)
        return np.sqrt(np.mean(audio_data**2))

    def is_speech(self, audio_frame):
        """
        Determine if audio frame contains speech using both VAD and energy
        """
        # Energy-based detection
        energy = self.calculate_energy(audio_frame)
        energy_speech = energy > self.energy_threshold
        
        # WebRTC VAD detection (requires specific frame size)
        try:
            vad_speech = self.vad.is_speech(audio_frame, self.RATE)
        except:
            vad_speech = False
        
        # Combine both methods (OR logic)
        return energy_speech or vad_speech

    def record_with_vad(self, max_duration=30):
        """
        Record audio with Voice Activity Detection
        Returns the recorded audio data or None if no speech detected
        """
        audio = pyaudio.PyAudio()
        
        try:
            stream = audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            self._update_status("🎤 Listening for speech...")
            
            # States
            is_recording = False
            speech_frames = []
            silence_counter = 0
            speech_counter = 0
            max_frames = int(self.RATE / self.CHUNK * max_duration)
            
            # Buffers
            silence_threshold_frames = int(self.max_silence_duration * self.RATE / self.CHUNK)
            speech_threshold_frames = int(self.min_speech_duration * self.RATE / self.CHUNK)
            
            for frame_count in range(max_frames):
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                
                # Check if current frame contains speech
                has_speech = self.is_speech(data)
                
                if has_speech:
                    if not is_recording:
                        self._update_status("🗣️ Speech detected, recording...")
                        is_recording = True
                    
                    speech_frames.append(data)
                    speech_counter += 1
                    silence_counter = 0
                    
                else:  # No speech
                    if is_recording:
                        silence_counter += 1
                        speech_frames.append(data)  # Keep recording during brief silences
                        
                        # Check if silence duration exceeded threshold
                        if silence_counter >= silence_threshold_frames:
                            self._update_status("⏹️ Speech ended, processing...")
                            break
                    else:
                        # Reset counters if we're not recording
                        speech_counter = 0
                        silence_counter = 0
            
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
        
        # Check if we have enough speech
        if speech_counter < speech_threshold_frames:
            self._update_status("Not enough speech detected")
            return None
        
        # Convert frames to audio data
        if speech_frames:
            audio_data = np.frombuffer(b"".join(speech_frames), dtype=np.int16)
            return audio_data
        
        return None

    def save_audio(self, audio_data, filename=None):
        """Save audio numpy array to file"""
        filename = filename or self.OUTPUT_FILENAME
        sf.write(filename, audio_data, self.RATE, subtype='PCM_16')

    def denoise_audio(self, audio_data):
        """
        Apply denoising to audio data using comprehensive denoising pipeline
        """
        if not self.enable_denoising:
            return audio_data
            
        self._update_status("🔧 Applying audio denoising...")
        
        try:
            # Convert int16 to float for processing
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data.astype(np.float32)
            
            # Apply comprehensive denoising
            denoised_audio = comprehensive_denoise(audio_float, self.RATE, method='all')
            
            # Convert back to int16
            denoised_audio_int16 = (denoised_audio * 32767).astype(np.int16)
            
            self._update_status("✅ Denoising completed")
            return denoised_audio_int16
            
        except Exception as e:
            self._update_status(f"Denoising error: {e}, using original audio")
            return audio_data

    def transcribe_audio(self, audio_path=None):
        """Transcribe audio file to text"""
        audio_path = audio_path or self.OUTPUT_FILENAME
        
        self._update_status("🔄 Transcribing audio...")
        output = self.transcriber(audio_path)
        return output["text"]

    def correct_with_llm(self, original_text):
        """
        Use LLM to correct STT output and ensure it's Vietnamese/English only
        """
        self._update_status("🤖 Correcting with LLM...")
        
        system_prompt = """You are a text correction assistant for a coffee shop speech recognition system. Your tasks:

1. Correct any speech recognition errors in the input text
2. Ensure the output is ONLY in Vietnamese or English (or both mixed)
3. If the text appears to be in any other language, respond with exactly: "UNSUPPORTED_LANGUAGE"
4. Focus on coffee shop context - correct common coffee terminology mistakes
5. Keep the meaning and intent of the original text
6. Return only the corrected text, no explanations
7. Also, you need to modify not just the ordering, but also greetings from the customer

Examples:
- "một cappuccino" → "một cappuccino" (correct Vietnamese)
- "I want latte" → "I want a latte" (add missing article)
- "cap-uh-chino" → "cappuccino" (fix pronunciation errors)
- "americano medium" → "americano medium" (correct as is)
- "我想要咖啡" → "UNSUPPORTED_LANGUAGE" (Chinese not supported)
- "Tên tôi là Duy" → "Tên tôi là Duy" (correct Vietnamese greeting)
- "Cho tôi một cốc mắt cha đã say" → "Cho tôi một cốc matcha đá xay"
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": original_text}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            corrected_text = response.choices[0].message.content.strip()
            
            # Check if language is unsupported
            if corrected_text == "UNSUPPORTED_LANGUAGE":
                return "I cannot understand, can you speak again?/Tôi không hiểu ý bạn, bạn có thể nói lại được không?"
            
            return corrected_text
            
        except Exception as e:
            self._update_status(f"LLM correction error: {e}")
            return original_text  # Return original if correction fails

    def process_single_recording(self, max_duration=30):
        """
        Process a single recording session and return results
        Returns tuple: (raw_text, corrected_text) or (None, None) if no speech
        """
        # Record with VAD
        audio_data = self.record_with_vad(max_duration=max_duration)
        
        if audio_data is None:
            return None, None
        
        # Apply denoising if enabled
        denoised_audio = self.denoise_audio(audio_data)
        
        # Save processed audio for transcription
        final_filename = self.DENOISED_FILENAME if self.enable_denoising else self.OUTPUT_FILENAME
        self.save_audio(denoised_audio, final_filename)
        
        # Transcribe using the processed audio
        raw_text = self.transcribe_audio(final_filename)
        
        # Correct with LLM
        corrected_text = self.correct_with_llm(raw_text)
        
        return raw_text, corrected_text


def main():
    """Main function for testing"""
    def status_update(message):
        print(f"Status: {message}")
    
    print("STT Module with Integrated Denoising Test")
    print("=" * 50)
    
    # Test with denoising enabled
    print("Testing with denoising enabled...")
    stt = STT_module(
        model_name="openai/whisper-medium", 
        enable_denoising=True,
        status_callback=status_update
    )
    
    # Test single recording
    raw_text, corrected_text = stt.process_single_recording()
    
    if raw_text:
        print(f"\n📝 Raw transcription: {raw_text}")
        print(f"🤖 LLM Corrected: {corrected_text}")
        print(f"\n📁 Denoised audio saved as: output_denoised.wav")
    else:
        print("❌ No speech detected")


if __name__ == "__main__":
    main()
