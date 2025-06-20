import os
import json
import datetime
from pathlib import Path
import soundfile as sf
from stt_whisper import STT_module

class CoffeeShopDatasetCreator:
    """
    Interactive dataset creator for coffee shop speech recognition evaluation
    Uses existing STT_module for recording and collects ground truth transcripts
    """
    
    def __init__(self, dataset_root="coffee_shop_dataset"):
        self.dataset_root = Path(dataset_root)
        self.audio_dir = self.dataset_root / "audio"
        self.transcripts_dir = self.dataset_root / "transcripts"
        
        # Create directories
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize STT module for recording
        print("Initializing STT module for recording...")
        self.stt = STT_module(status_callback=self.status_callback)
        
        # Load or create ground truth file
        self.ground_truth_file = self.transcripts_dir / "ground_truth.json"
        self.ground_truth = self.load_ground_truth()
        
        # Load or create metadata
        self.metadata_file = self.dataset_root / "metadata.json"
        self.metadata = self.load_metadata()
        
        print(f"Dataset directory: {self.dataset_root.absolute()}")
        print(f"Current dataset size: {len(self.ground_truth)} samples")
    
    def status_callback(self, message):
        """Status callback for STT module"""
        print(f"[STT] {message}")
    
    def load_ground_truth(self):
        """Load existing ground truth data or create new"""
        if self.ground_truth_file.exists():
            with open(self.ground_truth_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_ground_truth(self):
        """Save ground truth data to file"""
        with open(self.ground_truth_file, 'w', encoding='utf-8') as f:
            json.dump(self.ground_truth, f, ensure_ascii=False, indent=2)
    
    def load_metadata(self):
        """Load existing metadata or create new"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "dataset_info": {
                "created_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "total_samples": 0,
                "languages": ["vietnamese", "english", "mixed"],
                "audio_format": "16kHz_mono_wav",
                "categories": ["greetings", "orders", "questions", "other"]
            },
            "statistics": {
                "vietnamese_samples": 0,
                "english_samples": 0,
                "mixed_samples": 0,
                "greetings": 0,
                "orders": 0,
                "questions": 0,
                "other": 0
            }
        }
    
    def save_metadata(self):
        """Save metadata to file"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def get_next_filename(self):
        """Generate next available filename"""
        existing_files = list(self.audio_dir.glob("order_*.wav"))
        if not existing_files:
            return "order_001.wav"
        
        # Extract numbers and find the next one
        numbers = []
        for file in existing_files:
            try:
                num = int(file.stem.split('_')[1])
                numbers.append(num)
            except (IndexError, ValueError):
                continue
        
        next_num = max(numbers) + 1 if numbers else 1
        return f"order_{next_num:03d}.wav"
    
    def record_sample(self, max_duration=30):
        """Record a single audio sample"""
        print("\n" + "="*50)
        print("🎤 RECORDING NEW SAMPLE")
        print("="*50)
        
        # Get filename
        filename = self.get_next_filename()
        audio_path = self.audio_dir / filename
        
        print(f"Recording will be saved as: {filename}")
        input("Press Enter when ready to start recording...")
        
        # Record audio using STT module
        audio_data = self.stt.record_with_vad(max_duration=max_duration)
        
        if audio_data is None:
            print("❌ No speech detected. Try again.")
            return False
        
        # Save audio file
        sf.write(audio_path, audio_data, self.stt.RATE, subtype='PCM_16')
        print(f"✅ Audio saved: {audio_path}")
        
        # Collect ground truth from user (without showing STT predictions to avoid bias)
        print("\n📝 ENTER GROUND TRUTH TRANSCRIPT")
        print("-" * 30)
        print("Enter the correct transcription of what was actually said:")
        print("(This will be used as the reference for evaluation)")
        
        while True:
            reference = input("Ground truth: ").strip()
            if reference:
                break
            print("Please enter a non-empty transcription.")
        
        # Get additional metadata
        print("\n📊 ADDITIONAL METADATA")
        print("-" * 20)
        
        # Language detection
        print("Language options: 1=Vietnamese, 2=English, 3=Mixed")
        while True:
            try:
                lang_choice = int(input("Language (1/2/3): "))
                if lang_choice in [1, 2, 3]:
                    language = ["vietnamese", "english", "mixed"][lang_choice - 1]
                    break
                else:
                    print("Please enter 1, 2, or 3.")
            except ValueError:
                print("Please enter a number (1, 2, or 3).")
        
        # Category detection
        print("Category options: 1=Greetings, 2=Orders, 3=Questions, 4=Other")
        while True:
            try:
                cat_choice = int(input("Category (1/2/3/4): "))
                if cat_choice in [1, 2, 3, 4]:
                    category = ["greetings", "orders", "questions", "other"][cat_choice - 1]
                    break
                else:
                    print("Please enter 1, 2, 3, or 4.")
            except ValueError:
                print("Please enter a number (1, 2, 3, or 4).")
        
        # Speaker info (optional)
        speaker_info = input("Speaker info (optional, e.g., 'customer_1'): ").strip()
        if not speaker_info:
            speaker_info = f"speaker_{len(self.ground_truth) + 1}"
        
        # Get STT predictions in background for later analysis (not shown to avoid bias)
        print("\n🔄 Processing audio for analysis...")
        try:
            raw_text = self.stt.transcribe_audio(str(audio_path))
            corrected_text = self.stt.correct_with_llm(raw_text)
        except Exception as e:
            print(f"Note: STT processing failed: {e}")
            raw_text = "N/A"
            corrected_text = "N/A"
        
        # Save ground truth data
        self.ground_truth[filename] = {
            "reference": reference,
            "language": language,
            "category": category,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "speaker_info": speaker_info,
            "stt_raw": raw_text,
            "stt_corrected": corrected_text
        }
        
        # Update metadata statistics
        self.metadata["statistics"][f"{language}_samples"] += 1
        self.metadata["statistics"][category] += 1
        self.metadata["dataset_info"]["total_samples"] = len(self.ground_truth)
        
        # Save files
        self.save_ground_truth()
        self.save_metadata()
        
        print(f"\n✅ Sample '{filename}' added to dataset!")
        print(f"Total samples: {len(self.ground_truth)}")
        
        return True
    
    def show_statistics(self):
        """Display current dataset statistics"""
        print("\n📊 DATASET STATISTICS")
        print("="*30)
        print(f"Total samples: {self.metadata['dataset_info']['total_samples']}")
        print(f"Created: {self.metadata['dataset_info']['created_date']}")
        print(f"Dataset location: {self.dataset_root.absolute()}")
        
        print("\nLanguage distribution:")
        stats = self.metadata["statistics"]
        for lang in ["vietnamese", "english", "mixed"]:
            count = stats.get(f"{lang}_samples", 0)
            print(f"  {lang.capitalize()}: {count}")
        
        print("\nCategory distribution:")
        for cat in ["greetings", "orders", "questions", "other"]:
            count = stats.get(cat, 0)
            print(f"  {cat.capitalize()}: {count}")
        
        print(f"\nAudio files location: {self.audio_dir}")
        print(f"Ground truth file: {self.ground_truth_file}")
    
    def list_samples(self, limit=10):
        """List recent samples"""
        print(f"\n📋 RECENT SAMPLES (showing last {limit})")
        print("="*50)
        
        items = list(self.ground_truth.items())[-limit:]
        for filename, data in items:
            print(f"File: {filename}")
            print(f"  Reference: {data['reference']}")
            print(f"  Language: {data['language']}")
            print(f"  Category: {data['category']}")
            print(f"  Time: {data['timestamp']}")
            print()
    
    def run_interactive_session(self):
        """Run interactive dataset creation session"""
        print("\n🎯 COFFEE SHOP DATASET CREATOR")
        print("="*40)
        print("This tool helps you create a dataset for WER evaluation.")
        print("You'll record audio samples and provide ground truth transcripts.")
        
        while True:
            print("\n📋 MENU OPTIONS")
            print("-" * 20)
            print("1. Record new sample")
            print("2. Show statistics")
            print("3. List recent samples")
            print("4. Exit")
            
            try:
                choice = int(input("\nSelect option (1-4): "))
                
                if choice == 1:
                    self.record_sample()
                elif choice == 2:
                    self.show_statistics()
                elif choice == 3:
                    self.list_samples()
                elif choice == 4:
                    print("💾 Saving data...")
                    self.save_ground_truth()
                    self.save_metadata()
                    print("✅ Dataset creation session ended.")
                    print(f"Dataset saved to: {self.dataset_root.absolute()}")
                    break
                else:
                    print("Please enter a number between 1-4.")
                    
            except ValueError:
                print("Please enter a valid number.")
            except KeyboardInterrupt:
                print("\n\n💾 Saving data before exit...")
                self.save_ground_truth()
                self.save_metadata()
                print("✅ Data saved. Goodbye!")
                break


def main():
    """Main function to run the dataset creator"""
    print("Initializing Coffee Shop Dataset Creator...")
    
    # Ask for custom dataset location
    dataset_path = input("Enter dataset directory name (or press Enter for 'coffee_shop_dataset'): ").strip()
    if not dataset_path:
        dataset_path = "coffee_shop_dataset"
    
    # Create and run dataset creator
    creator = CoffeeShopDatasetCreator(dataset_path)
    creator.run_interactive_session()


if __name__ == "__main__":
    main() 