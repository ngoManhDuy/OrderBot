import pyaudio
import wave
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoFeatureExtractor
import noisereduce as nr
import numpy as np

# Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1  # Mono
RATE = 16000
CHUNK = 1024
OUTPUT_FILENAME = "output.wav"

# Load the model, tokenizer, and feature extractor separately
model = AutoModelForSpeechSeq2Seq.from_pretrained("vinai/PhoWhisper-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/PhoWhisper-base")
feature_extractor = AutoFeatureExtractor.from_pretrained("vinai/PhoWhisper-base")

# Initialize the ASR pipeline with model, tokenizer, and feature extractor
transcriber = pipeline("automatic-speech-recognition", model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)

def record_audio(duration, output_filename):
    audio = pyaudio.PyAudio()

    # Start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("Recording...")

    frames = []

    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished.")

    # Stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Convert frames to numpy array
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

    # Apply noise reduction
    reduced_noise = nr.reduce_noise(y=audio_data, sr=RATE)

    # Save the reduced noise audio to a file
    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(reduced_noise.tobytes())
    wf.close()

    print(f"Audio saved as {output_filename}")

def transcribe_audio(audio_path):
    output = transcriber(audio_path)
    return output['text']

if __name__ == "__main__":
    print("Please speak...")
    record_audio(5, OUTPUT_FILENAME)
    text = transcribe_audio(OUTPUT_FILENAME)
    print("Transcribed Text:", text)