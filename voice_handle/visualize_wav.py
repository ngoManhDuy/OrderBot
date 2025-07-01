import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.io.wavfile as wav
import argparse
import os
from scipy import signal

def load_audio(file_path):
    """Load audio file using librosa for better compatibility"""
    try:
        # Load with librosa (handles various formats better)
        y, sr = librosa.load(file_path, sr=None)
        return y, sr
    except Exception as e:
        print(f"Error loading with librosa: {e}")
        # Fallback to scipy
        try:
            sr, y = wav.read(file_path)
            # Normalize if integer format
            if y.dtype == np.int16:
                y = y / 32768.0
            elif y.dtype == np.int32:
                y = y / 2147483648.0
            return y, sr
        except Exception as e2:
            print(f"Error loading with scipy: {e2}")
            raise

def plot_waveform(y, sr, title="Audio Waveform"):
    """Plot the audio waveform"""
    plt.figure(figsize=(12, 4))
    time = np.linspace(0, len(y) / sr, len(y))
    plt.plot(time, y)
    plt.title(title)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def plot_spectrogram(y, sr, title="Spectrogram"):
    """Plot spectrogram using librosa"""
    plt.figure(figsize=(12, 6))
    
    # Compute spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    
    # Plot
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(label='dB')
    plt.title(title)
    plt.tight_layout()

def plot_mel_spectrogram(y, sr, title="Mel Spectrogram"):
    """Plot mel spectrogram"""
    plt.figure(figsize=(12, 6))
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
    
    # Plot
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(label='dB')
    plt.title(title)
    plt.tight_layout()

def plot_chromagram(y, sr, title="Chromagram"):
    """Plot chromagram (pitch class profiles)"""
    plt.figure(figsize=(12, 6))
    
    # Compute chromagram
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # Plot
    librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()

def plot_spectral_features(y, sr):
    """Plot various spectral features"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Spectral centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    times = librosa.times_like(cent, sr=sr)
    axes[0, 0].plot(times, cent)
    axes[0, 0].set_title('Spectral Centroid')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Hz')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    axes[0, 1].plot(times, rolloff)
    axes[0, 1].set_title('Spectral Rolloff')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Hz')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    axes[1, 0].plot(times, zcr)
    axes[1, 0].set_title('Zero Crossing Rate')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Rate')
    axes[1, 0].grid(True, alpha=0.3)
    
    # MFCC (first coefficient)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    axes[1, 1].plot(times, mfccs[0])
    axes[1, 1].set_title('MFCC (1st coefficient)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Coefficient value')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()

def analyze_audio_properties(y, sr, filename):
    """Print audio file properties and statistics"""
    duration = len(y) / sr
    
    print(f"\n=== Audio Analysis for {filename} ===")
    print(f"Sample Rate: {sr} Hz")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Total samples: {len(y)}")
    print(f"Channels: {'Mono' if len(y.shape) == 1 else 'Stereo'}")
    print(f"Min amplitude: {np.min(y):.4f}")
    print(f"Max amplitude: {np.max(y):.4f}")
    print(f"RMS amplitude: {np.sqrt(np.mean(y**2)):.4f}")
    
    # Frequency domain analysis
    fft = np.fft.fft(y)
    magnitude = np.abs(fft)
    freqs = np.fft.fftfreq(len(fft), 1/sr)
    
    # Find dominant frequency
    peak_freq_idx = np.argmax(magnitude[:len(magnitude)//2])
    peak_freq = freqs[peak_freq_idx]
    
    print(f"Dominant frequency: {peak_freq:.1f} Hz")
    print("=" * 50)

def visualize_wav_file(file_path, show_all=False):
    """Main function to visualize WAV file"""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found!")
        return
    
    try:
        # Load audio
        print(f"Loading audio file: {file_path}")
        y, sr = load_audio(file_path)
        
        # Analyze properties
        analyze_audio_properties(y, sr, os.path.basename(file_path))
        
        # Create visualizations
        print("Creating visualizations...")
        
        # Waveform
        plot_waveform(y, sr, f"Waveform - {os.path.basename(file_path)}")
        
        if show_all:
            # Spectrogram
            plot_spectrogram(y, sr, f"Spectrogram - {os.path.basename(file_path)}")
            
            # Mel Spectrogram
            plot_mel_spectrogram(y, sr, f"Mel Spectrogram - {os.path.basename(file_path)}")
            
            # Chromagram
            plot_chromagram(y, sr, f"Chromagram - {os.path.basename(file_path)}")
            
            # Spectral features
            plot_spectral_features(y, sr)
        else:
            # Just show spectrogram for basic visualization
            plot_spectrogram(y, sr, f"Spectrogram - {os.path.basename(file_path)}")
        
        # Show all plots
        plt.show()
        
    except Exception as e:
        print(f"Error processing file: {e}")

def compare_audio_files(file1, file2, title1="Original", title2="Processed"):
    """Compare two audio files side by side"""
    try:
        # Load both files
        y1, sr1 = load_audio(file1)
        y2, sr2 = load_audio(file2)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Waveforms
        time1 = np.linspace(0, len(y1) / sr1, len(y1))
        time2 = np.linspace(0, len(y2) / sr2, len(y2))
        
        axes[0, 0].plot(time1, y1)
        axes[0, 0].set_title(f'Waveform - {title1}')
        axes[0, 0].set_xlabel('Time (seconds)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(time2, y2)
        axes[0, 1].set_title(f'Waveform - {title2}')
        axes[0, 1].set_xlabel('Time (seconds)')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Spectrograms
        D1 = librosa.amplitude_to_db(np.abs(librosa.stft(y1)), ref=np.max)
        D2 = librosa.amplitude_to_db(np.abs(librosa.stft(y2)), ref=np.max)
        
        im1 = librosa.display.specshow(D1, sr=sr1, x_axis='time', y_axis='hz', ax=axes[1, 0])
        axes[1, 0].set_title(f'Spectrogram - {title1}')
        
        im2 = librosa.display.specshow(D2, sr=sr2, x_axis='time', y_axis='hz', ax=axes[1, 1])
        axes[1, 1].set_title(f'Spectrogram - {title2}')
        
        plt.tight_layout()
        plt.show()
        
        # Print comparison stats
        print(f"\n=== Audio Comparison ===")
        print(f"{title1}: RMS={np.sqrt(np.mean(y1**2)):.4f}, Peak={np.max(np.abs(y1)):.4f}")
        print(f"{title2}: RMS={np.sqrt(np.mean(y2**2)):.4f}, Peak={np.max(np.abs(y2)):.4f}")
        
    except Exception as e:
        print(f"Error comparing files: {e}")

def main():
    parser = argparse.ArgumentParser(description='Visualize WAV audio files')
    parser.add_argument('file', nargs='?', default='output.wav', 
                       help='WAV file to visualize (default: output.wav)')
    parser.add_argument('--all', '-a', action='store_true', 
                       help='Show all visualization types')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available WAV files in current directory')
    parser.add_argument('--compare', '-c', 
                       help='Compare with another WAV file (show both side by side)')
    
    args = parser.parse_args()
    
    if args.list:
        wav_files = [f for f in os.listdir('.') if f.lower().endswith('.wav')]
        if wav_files:
            print("Available WAV files:")
            for f in wav_files:
                print(f"  - {f}")
        else:
            print("No WAV files found in current directory")
        return
    
    if args.compare:
        if os.path.exists(args.compare):
            compare_audio_files(args.file, args.compare, 
                              os.path.basename(args.file), 
                              os.path.basename(args.compare))
        else:
            print(f"Comparison file '{args.compare}' not found!")
        return
    
    visualize_wav_file(args.file, args.all)

if __name__ == "__main__":
    main()
