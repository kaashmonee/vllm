#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class PatentClassificationVisualizer:
    def __init__(self, results_file):
        self.results_file = Path(results_file)
        with open(self.results_file, 'r') as f:
            self.data = json.load(f)
        
        self.class_names = self.data['experiment_metadata']['class_names']
        self.detailed_results = self.data['detailed_results']
        
    def plot_accuracy_comparison(self, save_path=None):
        """Plot accuracy across different few-shot configurations"""
        few_shot_counts = []
        accuracies = []
        processing_times = []
        
        for result in self.detailed_results:
            few_shot_counts.append(result['few_shot_count'])
            accuracies.append(result['overall_accuracy'])
            processing_times.append(result['processing_time_seconds'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy plot
        bars1 = ax1.bar(few_shot_counts, accuracies, alpha=0.8, color='skyblue', edgecolor='navy')
        ax1.set_xlabel('Few-shot Examples')
        ax1.set_ylabel('Overall Accuracy')
        ax1.set_title('Classification Accuracy vs Few-shot Examples')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Processing time plot
        bars2 = ax2.bar(few_shot_counts, processing_times, alpha=0.8, color='lightcoral', edgecolor='darkred')
        ax2.set_xlabel('Few-shot Examples')
        ax2.set_ylabel('Processing Time (seconds)')
        ax2.set_title('Processing Time vs Few-shot Examples')
        
        # Add value labels on bars
        for bar, time in zip(bars2, processing_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_confusion_matrices(self, save_path=None):
        """Plot confusion matrices for each few-shot configuration"""
        n_configs = len(self.detailed_results)
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, result in enumerate(self.detailed_results):
            confusion_matrix = np.array(result['confusion_matrix'])
            
            # Normalize confusion matrix
            cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm_normalized, 
                       annot=True, 
                       fmt='.2f', 
                       cmap='Blues',
                       xticklabels=list(self.class_names.keys()),
                       yticklabels=list(self.class_names.keys()),
                       ax=axes[i],
                       cbar_kws={'label': 'Normalized Count'})
            
            axes[i].set_title(f'{result["few_shot_count"]} Few-shot Examples\n'
                            f'Accuracy: {result["overall_accuracy"]:.3f}')
            axes[i].set_xlabel('Predicted Class')
            axes[i].set_ylabel('True Class')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_per_class_performance(self, save_path=None):
        """Plot F1 scores for each class across different configurations"""
        class_performance = {}
        
        # Collect F1 scores for each class across configurations
        for class_id, class_name in self.class_names.items():
            f1_scores = []
            for result in self.detailed_results:
                if class_id in result['per_class_metrics']:
                    f1_scores.append(result['per_class_metrics'][class_id]['f1'])
                else:
                    f1_scores.append(0.0)
            class_performance[class_name] = f1_scores
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame(class_performance, 
                         index=[f"{r['few_shot_count']}-shot" for r in self.detailed_results])
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(16, 8))
        sns.heatmap(df.T, annot=True, fmt='.3f', cmap='RdYlGn', 
                   ax=ax, cbar_kws={'label': 'F1 Score'})
        
        ax.set_title('Per-Class F1 Scores Across Few-shot Configurations')
        ax.set_xlabel('Few-shot Configuration')
        ax.set_ylabel('Patent Class')
        
        # Rotate y-axis labels for better readability
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return df
        
    def plot_class_distribution(self, save_path=None):
        """Plot class distribution in the dataset"""
        class_dist = self.data['experiment_metadata']['dataset_info']['class_distribution']
        
        # Convert to lists for plotting
        classes = []
        counts = []
        for class_id, count in class_dist.items():
            classes.append(f"Class {class_id}\n{self.class_names[class_id]}")
            counts.append(count)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        bars = ax.bar(range(len(classes)), counts, alpha=0.8, color='lightgreen', edgecolor='darkgreen')
        
        ax.set_xlabel('Patent Classes')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Class Distribution in Patent Dataset')
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                   str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def analyze_problem_areas(self):
        """Identify and analyze the most problematic classes"""
        print("=== PATENT CLASSIFICATION ANALYSIS ===\n")
        
        # Overall performance summary
        best_config = self.data['results_summary']['best_few_shot_config']
        worst_config = self.data['results_summary']['worst_few_shot_config']
        
        print(f"📊 OVERALL PERFORMANCE:")
        print(f"   • Best: {best_config['few_shot_count']}-shot with {best_config['accuracy']:.3f} accuracy")
        print(f"   • Worst: {worst_config['few_shot_count']}-shot with {worst_config['accuracy']:.3f} accuracy")
        print(f"   • Improvement: {(best_config['accuracy'] - worst_config['accuracy']):.3f}")
        print()
        
        # Per-class analysis
        per_class = self.data['analysis']['per_class_performance']
        
        print("🎯 PER-CLASS PERFORMANCE (sorted by F1 score):")
        class_f1_scores = [(class_id, info['mean_f1'], info['class_name']) 
                          for class_id, info in per_class.items()]
        class_f1_scores.sort(key=lambda x: x[1], reverse=True)
        
        for class_id, mean_f1, class_name in class_f1_scores:
            status = "✅" if mean_f1 > 0.5 else "❌" if mean_f1 < 0.3 else "⚠️"
            print(f"   {status} Class {class_id}: {mean_f1:.3f} F1 - {class_name}")
        print()
        
        # Problem identification
        problem_classes = [x for x in class_f1_scores if x[1] < 0.3]
        if problem_classes:
            print("🚨 MAJOR PROBLEM CLASSES (F1 < 0.3):")
            for class_id, mean_f1, class_name in problem_classes:
                class_info = per_class[class_id]
                print(f"   • Class {class_id} ({class_name}):")
                print(f"     - F1: {mean_f1:.3f} (std: {class_info['f1_std']:.3f})")
                print(f"     - Precision: {class_info['mean_precision']:.3f}")
                print(f"     - Recall: {class_info['mean_recall']:.3f}")
                print(f"     - Sample count: {class_info['mean_support']:.0f}")
            print()
        
        # Class imbalance analysis
        class_dist = self.data['experiment_metadata']['dataset_info']['class_distribution']
        total_samples = sum(class_dist.values())
        
        print("⚖️ CLASS IMBALANCE ANALYSIS:")
        imbalance_ratios = {}
        for class_id, count in class_dist.items():
            ratio = count / total_samples
            imbalance_ratios[class_id] = ratio
            class_name = self.class_names[class_id]
            status = "📈" if ratio > 0.15 else "📉" if ratio < 0.05 else "📊"
            print(f"   {status} Class {class_id}: {ratio:.1%} ({count} samples) - {class_name}")
        
        # Identify severely underrepresented classes
        underrepresented = [(k, v) for k, v in imbalance_ratios.items() if v < 0.05]
        if underrepresented:
            print(f"\n⚠️  SEVERELY UNDERREPRESENTED CLASSES (< 5% of data):")
            for class_id, ratio in underrepresented:
                print(f"   • Class {class_id}: {ratio:.1%} - {self.class_names[class_id]}")
        
        print(f"\n📋 RECOMMENDATIONS:")
        for rec in self.data['analysis']['recommendations']:
            print(f"   • {rec}")
        
    def generate_full_report(self, output_dir="visualization_output"):
        """Generate all visualizations and save to directory"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Generating comprehensive visualization report in: {output_path}")
        
        # Generate all plots
        self.plot_accuracy_comparison(output_path / "accuracy_comparison.png")
        self.plot_confusion_matrices(output_path / "confusion_matrices.png")
        performance_df = self.plot_per_class_performance(output_path / "per_class_performance.png")
        self.plot_class_distribution(output_path / "class_distribution.png")
        
        # Save performance DataFrame as CSV
        performance_df.to_csv(output_path / "per_class_f1_scores.csv")
        
        # Generate text analysis
        self.analyze_problem_areas()
        
        print(f"\n✅ Full report generated! Check the '{output_dir}' directory.")

def main():
    parser = argparse.ArgumentParser(description='Visualize patent classification results')
    parser.add_argument('results_file', default='my_results.json', nargs='?',
                       help='Path to results JSON file (default: my_results.json)')
    parser.add_argument('--output-dir', default='visualization_output',
                       help='Output directory for visualizations')
    parser.add_argument('--analysis-only', action='store_true',
                       help='Only run text analysis, skip plots')
    
    args = parser.parse_args()
    
    if not Path(args.results_file).exists():
        print(f"❌ Results file not found: {args.results_file}")
        return
    
    visualizer = PatentClassificationVisualizer(args.results_file)
    
    if args.analysis_only:
        visualizer.analyze_problem_areas()
    else:
        visualizer.generate_full_report(args.output_dir)

if __name__ == "__main__":
    main()