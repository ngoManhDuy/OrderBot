import json
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jiwer import wer, cer
import numpy as np
from collections import defaultdict, Counter
import re
from typing import Dict, List, Tuple, Optional

class CoffeeShopWERAnalyzer:
    """
    Comprehensive WER analysis for Coffee Shop Speech Recognition System
    Evaluates both raw STT output and LLM-corrected output
    """
    
    def __init__(self, dataset_path="data"):
        self.dataset_path = Path(dataset_path)
        self.ground_truth_file = self.dataset_path / "transcripts" / "ground_truth.json"
        self.metadata_file = self.dataset_path / "metadata.json"
        self.audio_dir = self.dataset_path / "audio"
        
        # Load data
        self.ground_truth = self.load_ground_truth()
        self.metadata = self.load_metadata()
        
        # Results storage
        self.results = {}
        self.detailed_results = []
        
        print(f"Loaded dataset with {len(self.ground_truth)} samples")
        print(f"Dataset path: {self.dataset_path.absolute()}")
    
    def load_ground_truth(self):
        """Load ground truth data"""
        with open(self.ground_truth_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_metadata(self):
        """Load metadata"""
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for fair WER comparison
        - Convert to lowercase
        - Remove extra whitespace
        - Handle punctuation consistently
        """
        # Handle special LLM responses
        if "I cannot understand" in text or "Tôi không hiểu" in text:
            return "[UNSUPPORTED]"
        
        # Basic normalization
        text = text.lower().strip()
        # Remove leading/trailing punctuation but keep internal
        text = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def calculate_detailed_measures(self, reference: str, hypothesis: str) -> Dict:
        """
        Calculate detailed error measures manually
        """
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        # Simple alignment for basic measures
        ref_len = len(ref_words)
        hyp_len = len(hyp_words)
        
        # Basic approximation - for exact measures, would need proper alignment
        if ref_len == 0 and hyp_len == 0:
            return {'substitutions': 0, 'deletions': 0, 'insertions': 0, 'hits': 0}
        elif ref_len == 0:
            return {'substitutions': 0, 'deletions': 0, 'insertions': hyp_len, 'hits': 0}
        elif hyp_len == 0:
            return {'substitutions': 0, 'deletions': ref_len, 'insertions': 0, 'hits': 0}
        
        # Simple word-level comparison (not perfect alignment, but good approximation)
        common_words = set(ref_words) & set(hyp_words)
        hits = len(common_words)
        
        # Rough approximation of edit operations
        max_len = max(ref_len, hyp_len)
        min_len = min(ref_len, hyp_len)
        
        substitutions = min_len - hits
        if ref_len > hyp_len:
            deletions = ref_len - hyp_len
            insertions = 0
        else:
            deletions = 0
            insertions = hyp_len - ref_len
            
        return {
            'substitutions': max(0, substitutions),
            'deletions': max(0, deletions), 
            'insertions': max(0, insertions),
            'hits': hits
        }
    
    def calculate_wer_details(self, reference: str, hypothesis: str) -> Dict:
        """
        Calculate detailed WER metrics including error breakdown
        """
        ref_normalized = self.normalize_text(reference)
        hyp_normalized = self.normalize_text(hypothesis)
        
        # Handle special cases
        if hyp_normalized == "[UNSUPPORTED]":
            return {
                'wer': 1.0,  # 100% error for unsupported language
                'cer': 1.0,
                'substitutions': 0,
                'deletions': len(ref_normalized.split()),
                'insertions': 0,
                'hits': 0,
                'ref_words': len(ref_normalized.split()),
                'hyp_words': 0,
                'error_type': 'language_rejection'
            }
        
        try:
            # Calculate WER and CER
            wer_score = wer([ref_normalized], [hyp_normalized])
            cer_score = cer([ref_normalized], [hyp_normalized])
            
            # Get detailed measures using our custom function
            measures = self.calculate_detailed_measures(ref_normalized, hyp_normalized)
            
            return {
                'wer': wer_score,
                'cer': cer_score,
                'substitutions': measures['substitutions'],
                'deletions': measures['deletions'],
                'insertions': measures['insertions'],
                'hits': measures['hits'],
                'ref_words': len(ref_normalized.split()),
                'hyp_words': len(hyp_normalized.split()),
                'error_type': 'normal_processing'
            }
            
        except Exception as e:
            print(f"Error calculating WER for '{reference}' vs '{hypothesis}': {e}")
            return {
                'wer': 1.0, 'cer': 1.0, 'substitutions': 0, 'deletions': 0,
                'insertions': 0, 'hits': 0, 'ref_words': 0, 'hyp_words': 0,
                'error_type': 'calculation_error'
            }
    
    def analyze_errors_by_category(self) -> Dict:
        """
        Analyze errors by different categories
        """
        error_analysis = {
            'by_language': defaultdict(list),
            'by_category': defaultdict(list),
            'by_error_type': defaultdict(int),
            'coffee_terms': defaultdict(int),
            'common_errors': defaultdict(int)
        }
        
        # Coffee shop specific terms to track
        coffee_terms = [
            'cappuccino', 'americano', 'espresso', 'latte', 'macchiato',
            'cà phê', 'trà sữa', 'chân châu', 'bạc xỉu', 'matcha'
        ]
        
        for filename, data in self.ground_truth.items():
            if filename not in self.detailed_results:
                continue
                
            result = next(r for r in self.detailed_results if r['filename'] == filename)
            
            # Group by language
            error_analysis['by_language'][data['language']].append(result['raw_wer'])
            
            # Group by category  
            error_analysis['by_category'][data['category']].append(result['raw_wer'])
            
            # Track coffee term accuracy
            reference_lower = data['reference'].lower()
            for term in coffee_terms:
                if term in reference_lower:
                    raw_hyp_lower = data['stt_raw'].lower()
                    if term not in raw_hyp_lower:
                        error_analysis['coffee_terms'][term] += 1
            
            # Common error patterns
            if result['raw_wer'] > 0.5:  # High error rate
                error_analysis['common_errors']['high_wer_samples'] += 1
            
            if result['corrected_wer'] < result['raw_wer']:
                error_analysis['common_errors']['llm_improved'] += 1
            elif result['corrected_wer'] > result['raw_wer']:
                error_analysis['common_errors']['llm_degraded'] += 1
        
        return error_analysis
    
    def run_evaluation(self) -> Dict:
        """
        Run complete WER evaluation
        """
        print("\n🔄 Running WER Evaluation...")
        print("=" * 50)
        
        raw_wers = []
        corrected_wers = []
        raw_cers = []
        corrected_cers = []
        
        for filename, data in self.ground_truth.items():
            reference = data['reference']
            raw_hypothesis = data['stt_raw']
            corrected_hypothesis = data['stt_corrected']
            
            # Calculate WER for raw STT output
            raw_metrics = self.calculate_wer_details(reference, raw_hypothesis)
            raw_wers.append(raw_metrics['wer'])
            raw_cers.append(raw_metrics['cer'])
            
            # Calculate WER for LLM-corrected output
            corrected_metrics = self.calculate_wer_details(reference, corrected_hypothesis)
            corrected_wers.append(corrected_metrics['wer'])
            corrected_cers.append(corrected_metrics['cer'])
            
            # Store detailed results
            self.detailed_results.append({
                'filename': filename,
                'reference': reference,
                'raw_hypothesis': raw_hypothesis,
                'corrected_hypothesis': corrected_hypothesis,
                'language': data['language'],
                'category': data['category'],
                'raw_wer': raw_metrics['wer'],
                'corrected_wer': corrected_metrics['wer'],
                'raw_cer': raw_metrics['cer'],
                'corrected_cer': corrected_metrics['cer'],
                'improvement': raw_metrics['wer'] - corrected_metrics['wer'],
                'raw_details': raw_metrics,
                'corrected_details': corrected_metrics
            })
            
            # Print individual results
            print(f"\n📝 {filename}")
            print(f"Reference: {reference}")
            print(f"Raw STT: {raw_hypothesis}")
            print(f"Corrected: {corrected_hypothesis}")
            print(f"Raw WER: {raw_metrics['wer']:.3f} | Corrected WER: {corrected_metrics['wer']:.3f}")
            print(f"Improvement: {raw_metrics['wer'] - corrected_metrics['wer']:.3f}")
        
        # Calculate overall statistics
        self.results = {
            'overall': {
                'raw_wer_mean': np.mean(raw_wers),
                'raw_wer_std': np.std(raw_wers),
                'corrected_wer_mean': np.mean(corrected_wers),
                'corrected_wer_std': np.std(corrected_wers),
                'raw_cer_mean': np.mean(raw_cers),
                'corrected_cer_mean': np.mean(corrected_cers),
                'improvement_mean': np.mean(raw_wers) - np.mean(corrected_wers),
                'total_samples': len(raw_wers)
            }
        }
        
        # Add breakdown by language and category
        self.results['by_language'] = self.calculate_breakdown_stats('language')
        self.results['by_category'] = self.calculate_breakdown_stats('category')
        
        # Error analysis
        self.results['error_analysis'] = self.analyze_errors_by_category()
        
        return self.results
    
    def calculate_breakdown_stats(self, group_by: str) -> Dict:
        """Calculate statistics broken down by language or category"""
        breakdown = defaultdict(lambda: {'raw_wers': [], 'corrected_wers': [], 'improvements': []})
        
        for result in self.detailed_results:
            group = result[group_by]
            breakdown[group]['raw_wers'].append(result['raw_wer'])
            breakdown[group]['corrected_wers'].append(result['corrected_wer'])
            breakdown[group]['improvements'].append(result['improvement'])
        
        # Calculate statistics for each group
        stats = {}
        for group, data in breakdown.items():
            stats[group] = {
                'raw_wer_mean': np.mean(data['raw_wers']),
                'corrected_wer_mean': np.mean(data['corrected_wers']),
                'improvement_mean': np.mean(data['improvements']),
                'sample_count': len(data['raw_wers']),
                'raw_wer_std': np.std(data['raw_wers']),
                'corrected_wer_std': np.std(data['corrected_wers'])
            }
        
        return stats
    
    def print_summary(self):
        """Print comprehensive evaluation summary"""
        if not self.results:
            print("No evaluation results available. Run evaluation first.")
            return
        
        print("\n" + "="*60)
        print("📊 COFFEE SHOP STT SYSTEM - WER EVALUATION SUMMARY")
        print("="*60)
        
        # Overall Results
        overall = self.results['overall']
        print(f"\n🎯 OVERALL PERFORMANCE")
        print("-" * 30)
        print(f"Total Samples: {overall['total_samples']}")
        print(f"Raw STT WER: {overall['raw_wer_mean']:.3f} ± {overall['raw_wer_std']:.3f}")
        print(f"LLM Corrected WER: {overall['corrected_wer_mean']:.3f} ± {overall['corrected_wer_std']:.3f}")
        print(f"Average Improvement: {overall['improvement_mean']:.3f}")
        print(f"Raw STT CER: {overall['raw_cer_mean']:.3f}")
        print(f"LLM Corrected CER: {overall['corrected_cer_mean']:.3f}")
        
        # Language Breakdown
        print(f"\n🌐 PERFORMANCE BY LANGUAGE")
        print("-" * 30)
        for lang, stats in self.results['by_language'].items():
            print(f"{lang.capitalize()}:")
            print(f"  Samples: {stats['sample_count']}")
            print(f"  Raw WER: {stats['raw_wer_mean']:.3f}")
            print(f"  Corrected WER: {stats['corrected_wer_mean']:.3f}")
            print(f"  Improvement: {stats['improvement_mean']:.3f}")
        
        # Category Breakdown
        print(f"\n📋 PERFORMANCE BY CATEGORY")
        print("-" * 30)
        for cat, stats in self.results['by_category'].items():
            print(f"{cat.capitalize()}:")
            print(f"  Samples: {stats['sample_count']}")
            print(f"  Raw WER: {stats['raw_wer_mean']:.3f}")
            print(f"  Corrected WER: {stats['corrected_wer_mean']:.3f}")
            print(f"  Improvement: {stats['improvement_mean']:.3f}")
        
        # Best and Worst Performances
        print(f"\n🏆 BEST & WORST SAMPLES")
        print("-" * 30)
        
        # Sort by raw WER
        sorted_results = sorted(self.detailed_results, key=lambda x: x['raw_wer'])
        
        print("Best Raw STT Performance:")
        for result in sorted_results[:3]:
            print(f"  {result['filename']}: WER={result['raw_wer']:.3f} - {result['reference'][:50]}...")
        
        print("\nWorst Raw STT Performance:")
        for result in sorted_results[-3:]:
            print(f"  {result['filename']}: WER={result['raw_wer']:.3f} - {result['reference'][:50]}...")
        
        # LLM Impact
        llm_improvements = [r for r in self.detailed_results if r['improvement'] > 0]
        llm_degradations = [r for r in self.detailed_results if r['improvement'] < 0]
        
        print(f"\n🤖 LLM CORRECTION IMPACT")
        print("-" * 30)
        print(f"Samples improved by LLM: {len(llm_improvements)}/{len(self.detailed_results)}")
        print(f"Samples degraded by LLM: {len(llm_degradations)}/{len(self.detailed_results)}")
        
        if llm_improvements:
            best_improvement = max(llm_improvements, key=lambda x: x['improvement'])
            print(f"Best improvement: {best_improvement['improvement']:.3f} ({best_improvement['filename']})")
        
        if llm_degradations:
            worst_degradation = min(llm_degradations, key=lambda x: x['improvement'])
            print(f"Worst degradation: {worst_degradation['improvement']:.3f} ({worst_degradation['filename']})")
    
    def save_detailed_report(self, output_file="wer_detailed_report.csv"):
        """Save detailed results to CSV file"""
        if not self.detailed_results:
            print("No results to save. Run evaluation first.")
            return
        
        # Convert to DataFrame
        df_data = []
        for result in self.detailed_results:
            row = {
                'filename': result['filename'],
                'reference': result['reference'],
                'raw_hypothesis': result['raw_hypothesis'],
                'corrected_hypothesis': result['corrected_hypothesis'],
                'language': result['language'],
                'category': result['category'],
                'raw_wer': result['raw_wer'],
                'corrected_wer': result['corrected_wer'],
                'raw_cer': result['raw_cer'],
                'corrected_cer': result['corrected_cer'],
                'improvement': result['improvement'],
                'raw_substitutions': result['raw_details']['substitutions'],
                'raw_deletions': result['raw_details']['deletions'],
                'raw_insertions': result['raw_details']['insertions'],
                'corrected_substitutions': result['corrected_details']['substitutions'],
                'corrected_deletions': result['corrected_details']['deletions'],
                'corrected_insertions': result['corrected_details']['insertions']
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✅ Detailed report saved to: {output_file}")
        
        return df
    
    def create_visualizations(self, save_plots=True):
        """Create visualization plots for WER analysis"""
        if not self.detailed_results:
            print("No results to visualize. Run evaluation first.")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Coffee Shop STT System - WER Analysis', fontsize=16, fontweight='bold')
        
        # 1. Overall WER Comparison
        ax1 = axes[0, 0]
        raw_wers = [r['raw_wer'] for r in self.detailed_results]
        corrected_wers = [r['corrected_wer'] for r in self.detailed_results]
        
        x = np.arange(len(raw_wers))
        width = 0.35
        
        ax1.bar(x - width/2, raw_wers, width, label='Raw STT', alpha=0.8)
        ax1.bar(x + width/2, corrected_wers, width, label='LLM Corrected', alpha=0.8)
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Word Error Rate')
        ax1.set_title('WER Comparison: Raw vs Corrected')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. WER by Language
        ax2 = axes[0, 1]
        lang_data = self.results['by_language']
        languages = list(lang_data.keys())
        raw_means = [lang_data[lang]['raw_wer_mean'] for lang in languages]
        corrected_means = [lang_data[lang]['corrected_wer_mean'] for lang in languages]
        
        x = np.arange(len(languages))
        ax2.bar(x - width/2, raw_means, width, label='Raw STT', alpha=0.8)
        ax2.bar(x + width/2, corrected_means, width, label='LLM Corrected', alpha=0.8)
        ax2.set_xlabel('Language')
        ax2.set_ylabel('Mean WER')
        ax2.set_title('WER by Language')
        ax2.set_xticks(x)
        ax2.set_xticklabels(languages)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Improvement Distribution
        ax3 = axes[0, 2]
        improvements = [r['improvement'] for r in self.detailed_results]
        ax3.hist(improvements, bins=10, alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', label='No change')
        ax3.set_xlabel('WER Improvement (Raw - Corrected)')
        ax3.set_ylabel('Number of Samples')
        ax3.set_title('Distribution of LLM Improvements')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Sample-wise WER Trends
        ax4 = axes[1, 0]
        sample_indices = range(len(self.detailed_results))
        ax4.plot(sample_indices, raw_wers, 'o-', label='Raw STT', alpha=0.7)
        ax4.plot(sample_indices, corrected_wers, 's-', label='LLM Corrected', alpha=0.7)
        ax4.set_xlabel('Sample Index')
        ax4.set_ylabel('WER')
        ax4.set_title('WER Trends Across Samples')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Error Type Analysis
        ax5 = axes[1, 1]
        error_counts = {'Improved': 0, 'Degraded': 0, 'No Change': 0}
        for r in self.detailed_results:
            if r['improvement'] > 0.01:
                error_counts['Improved'] += 1
            elif r['improvement'] < -0.01:
                error_counts['Degraded'] += 1
            else:
                error_counts['No Change'] += 1
        
        ax5.pie(error_counts.values(), labels=error_counts.keys(), autopct='%1.1f%%', 
                startangle=90, colors=['green', 'red', 'gray'])
        ax5.set_title('LLM Correction Impact')
        
        # 6. WER vs CER Correlation
        ax6 = axes[1, 2]
        ax6.scatter([r['raw_wer'] for r in self.detailed_results], 
                   [r['raw_cer'] for r in self.detailed_results], 
                   alpha=0.7, label='Raw STT')
        ax6.scatter([r['corrected_wer'] for r in self.detailed_results], 
                   [r['corrected_cer'] for r in self.detailed_results], 
                   alpha=0.7, label='LLM Corrected')
        ax6.set_xlabel('Word Error Rate (WER)')
        ax6.set_ylabel('Character Error Rate (CER)')
        ax6.set_title('WER vs CER Correlation')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('wer_analysis_plots.png', dpi=300, bbox_inches='tight')
            print("✅ Plots saved to: wer_analysis_plots.png")
        
        plt.show()


def main():
    """Main function to run WER evaluation"""
    print("🎯 Coffee Shop STT System - WER Evaluation Tool")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = CoffeeShopWERAnalyzer("data")
    
    # Run evaluation
    results = analyzer.run_evaluation()
    
    # Print summary
    analyzer.print_summary()
    
    # Save detailed report
    df = analyzer.save_detailed_report()
    
    # Create visualizations
    analyzer.create_visualizations()
    
    print("\n✅ WER evaluation completed!")
    print(f"📊 Check 'wer_detailed_report.csv' for detailed results")
    print(f"📈 Check 'wer_analysis_plots.png' for visualizations")


if __name__ == "__main__":
    main() 