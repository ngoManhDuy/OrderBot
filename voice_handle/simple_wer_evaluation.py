import json
from pathlib import Path
from jiwer import wer
import re

class SimpleWERAnalyzer:

    def __init__(self, dataset_path="data"):
        self.dataset_path = Path(dataset_path)
        self.ground_truth_file = self.dataset_path / "transcripts" / "ground_truth.json"
        
        # Load data
        with open(self.ground_truth_file, 'r', encoding='utf-8') as f:
            self.ground_truth = json.load(f)
        
        print(f"Loaded {len(self.ground_truth)} samples")
    
    def normalize_text(self, text: str) -> str:
        """Simple text normalization for WER calculation"""
        # Handle LLM rejection responses
        if "I cannot understand" in text or "Tôi không hiểu" in text:
            return "[UNSUPPORTED]"
        
        # Basic normalization
        text = text.lower().strip()
        text = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text
    
    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """Calculate WER between reference and hypothesis"""
        ref_normalized = self.normalize_text(reference)
        hyp_normalized = self.normalize_text(hypothesis)
        
        # Handle unsupported language case
        if hyp_normalized == "[UNSUPPORTED]":
            return 1.0  # 100% error
        
        try:
            return wer([ref_normalized], [hyp_normalized])
        except:
            return 1.0  # Return 100% error if calculation fails
    
    def run_evaluation(self):
        """Run WER evaluation and print results"""
        print("WER EVALUATION")
        
        raw_wers = []
        corrected_wers = []
        improvements = []
        
        print(f"\n{'File':<15} {'Raw WER':<10} {'LLM WER':<10} {'Improve':<10} {'Reference'}")
        print("-" * 80)
        
        for filename, data in self.ground_truth.items():
            reference = data['reference']
            raw_hypothesis = data['stt_raw']
            corrected_hypothesis = data['stt_corrected']
            
            # Calculate WER
            raw_wer = self.calculate_wer(reference, raw_hypothesis)
            corrected_wer = self.calculate_wer(reference, corrected_hypothesis)
            improvement = raw_wer - corrected_wer
            
            # Store results
            raw_wers.append(raw_wer)
            corrected_wers.append(corrected_wer)
            improvements.append(improvement)
            
            # Print individual results
            print(f"{filename:<15} {raw_wer:<10.3f} {corrected_wer:<10.3f} {improvement:<+10.3f} {reference[:40]}...")
        
        # Calculate overall statistics
        print("OVERALL RESULTS")
        print(f"Total Samples: {len(raw_wers)}")
        print(f"Raw STT WER (mean): {sum(raw_wers)/len(raw_wers):.3f}")
        print(f"LLM Corrected WER (mean): {sum(corrected_wers)/len(corrected_wers):.3f}")
        print(f"Average Improvement: {sum(improvements)/len(improvements):.3f}")
        
        # Count improvements
        improved = sum(1 for imp in improvements if imp > 0.01)
        degraded = sum(1 for imp in improvements if imp < -0.01)
        unchanged = len(improvements) - improved - degraded
        
        print(f"\nLLM Impact:")
        print(f"  Improved: {improved}/{len(improvements)} samples")
        print(f"  Degraded: {degraded}/{len(improvements)} samples") 
        print(f"  Unchanged: {unchanged}/{len(improvements)} samples")
        
        # Best and worst samples
        results = list(zip(raw_wers, corrected_wers, improvements, self.ground_truth.keys()))
        results.sort(key=lambda x: x[0])  # Sort by raw WER
        
        print(f"\nBest Raw STT Performance:")
        for i in range(min(3, len(results))):
            raw_wer, corrected_wer, improvement, filename = results[i]
            print(f"  {filename}: {raw_wer:.3f} WER")
        
        print(f"\nWorst Raw STT Performance:")
        for i in range(max(0, len(results)-3), len(results)):
            raw_wer, corrected_wer, improvement, filename = results[i]
            print(f"  {filename}: {raw_wer:.3f} WER")
        
        print("\nEvaluation completed!")


def main():
    """Run simple WER evaluation"""
    print("Simple Coffee Shop STT WER Evaluation")
    
    analyzer = SimpleWERAnalyzer("data")
    analyzer.run_evaluation()


if __name__ == "__main__":
    main() 