"""
Ablation Study for Legal Assistance Platform

This script allows for systematic testing of the system with different components
enabled or disabled to measure their impact on overall performance.
"""
import sys
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path to import from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils import IPCEmbeddingManager

# Import necessary modules for ablation studies
try:
    # Import analyze_with_llm directly from utils
    from utils import analyze_with_llm
except ImportError:
    try:
        # Try fallback to app.py if utils doesn't have it
        from app import analyze_with_llm
    except ImportError:
        print("Warning: Could not import LLM analysis function. Some features may be unavailable.")


class AblationStudyManager:
    def __init__(self, test_cases_path=None):
        """
        Initialize the ablation study manager
        
        Args:
            test_cases_path: Path to test cases CSV file (optional)
        """
        self.embedding_manager = IPCEmbeddingManager()
        # Load the vector store on initialization
        try:
            self.embedding_manager.load_vector_store()
        except Exception as e:
            print(f"Warning: Could not load vector store: {e}")
        
        # Use our prepared test data
        if test_cases_path is None:
            test_cases_path = "../../data/test_data/ablation_test_cases.csv"
            
        # Load test cases if provided, otherwise use default ones
        if os.path.exists(test_cases_path):
            print(f"Loading test cases from: {test_cases_path}")
            self.test_cases = pd.read_csv(test_cases_path)
        else:
            print(f"Warning: Test cases file not found at {test_cases_path}. Using default test cases.")
            # Create simple test cases
            self.test_cases = pd.DataFrame({
                'id': range(1, 6),
                'description': [
                    "A person stole my mobile phone from my pocket while I was on a bus",
                    "My neighbor has been threatening me with violence if I don't sell my property to him",
                    "Someone hacked into my email account and sent inappropriate messages to my colleagues",
                    "A group of individuals have been regularly harassing me based on my religion",
                    "A company promised certain features in their product but failed to deliver as per contract"
                ],
                'ground_truth': [
                    "378,379,356",  # Theft, Criminal breach of trust, Criminal intimidation
                    "503,506,447",  # Criminal intimidation, Criminal intimidation, Criminal trespass
                    "66C,66D,419", # IT Act sections for identity theft and cheating by personation, Cheating
                    "295A,153A,506", # Deliberate acts to outrage religious feelings, Promoting enmity, Criminal intimidation
                    "420,406,415"  # Cheating, Criminal breach of trust, Cheating
                ]
            })
            
        self.results = {}
        
    def run_baseline(self, num_results=5):
        """
        Run the baseline system with all components enabled
        
        Args:
            num_results: Number of results to retrieve per case
        """
        print("Running baseline evaluation (all components enabled)...")
        
        precision_scores = []
        retrieval_times = []
        analysis_times = []
        total_times = []
        
        for _, case in self.test_cases.iterrows():
            description = case['description']
            ground_truth = set(case['ground_truth'].split(','))
            
            # Measure retrieval time
            retrieval_start = time.time()
            try:
                results = self.embedding_manager.search_similar_sections(description, k=num_results)
                retrieval_end = time.time()
            
                # Get retrieved sections
                retrieved_sections = []
                for doc, _ in results:
                    # Extract just the section number from the section field
                    # This assumes the section field is in a format like "Section 123" or similar
                    section = doc.metadata['section']
                    if isinstance(section, str) and section.startswith('Section '):
                        section_num = section.split(' ')[1]
                        retrieved_sections.append(section_num)
                    else:
                        retrieved_sections.append(str(section))
                
                # Calculate precision
                retrieved_set = set(retrieved_sections)
                relevant_retrieved = len(ground_truth.intersection(retrieved_set))
                precision = relevant_retrieved / len(retrieved_set) if retrieved_set else 0
            except Exception as e:
                print(f"Error retrieving results for '{description}': {e}")
                retrieval_end = time.time()
                precision = 0
                retrieved_sections = []
                results = []
            
            # Measure LLM analysis time if available
            try:
                formatted_results = [
                    {
                        'section': doc.metadata['section'], 
                        'content': doc.page_content,
                        'similarity': float(1 - score)  # Convert distance to similarity
                    } 
                    for doc, score in results
                ]
                
                analysis_start = time.time()
                _ = analyze_with_llm(description, formatted_results)
                analysis_end = time.time()
                analysis_time = analysis_end - analysis_start
            except Exception as e:
                # LLM analysis not available or failed
                print(f"LLM analysis unavailable or failed: {e}")
                analysis_time = 0
                
            # Record metrics
            retrieval_time = retrieval_end - retrieval_start
            total_time = retrieval_time + analysis_time
            
            precision_scores.append(precision)
            retrieval_times.append(retrieval_time)
            analysis_times.append(analysis_time)
            total_times.append(total_time)
        
        # Store baseline results
        self.results['baseline'] = {
            'precision': np.mean(precision_scores),
            'retrieval_time': np.mean(retrieval_times),
            'analysis_time': np.mean(analysis_times),
            'total_time': np.mean(total_times)
        }
        
        print(f"Baseline precision: {self.results['baseline']['precision']:.4f}")
        print(f"Baseline retrieval time: {self.results['baseline']['retrieval_time']:.4f} seconds")
        print(f"Baseline total time: {self.results['baseline']['total_time']:.4f} seconds")
        
        return self.results['baseline']
    
    def run_without_llm(self, num_results=5):
        """
        Run the system without LLM analysis
        
        Args:
            num_results: Number of results to retrieve per case
        """
        print("Running evaluation without LLM analysis...")
        
        precision_scores = []
        retrieval_times = []
        
        for _, case in self.test_cases.iterrows():
            description = case['description']
            ground_truth = set(case['ground_truth'].split(','))
            
            # Measure retrieval time
            retrieval_start = time.time()
            try:
                results = self.embedding_manager.search_similar_sections(description, k=num_results)
                retrieval_end = time.time()
                
                # Get retrieved sections
                retrieved_sections = []
                for doc, _ in results:
                    section = doc.metadata['section']
                    if isinstance(section, str) and section.startswith('Section '):
                        section_num = section.split(' ')[1]
                        retrieved_sections.append(section_num)
                    else:
                        retrieved_sections.append(str(section))
                
                # Calculate precision
                retrieved_set = set(retrieved_sections)
                relevant_retrieved = len(ground_truth.intersection(retrieved_set))
                precision = relevant_retrieved / len(retrieved_set) if retrieved_set else 0
            except Exception as e:
                print(f"Error retrieving results for '{description}': {e}")
                retrieval_end = time.time()
                precision = 0
            
            # Record metrics
            retrieval_time = retrieval_end - retrieval_start
            
            precision_scores.append(precision)
            retrieval_times.append(retrieval_time)
        
        # Store results
        self.results['without_llm'] = {
            'precision': np.mean(precision_scores),
            'retrieval_time': np.mean(retrieval_times),
            'analysis_time': 0,
            'total_time': np.mean(retrieval_times)
        }
        
        print(f"Precision without LLM: {self.results['without_llm']['precision']:.4f}")
        print(f"Retrieval time without LLM: {self.results['without_llm']['retrieval_time']:.4f} seconds")
        
        return self.results['without_llm']
    
    def run_smaller_results_set(self, num_results=3):
        """
        Run the system with a smaller result set
        
        Args:
            num_results: Number of results to retrieve per case
        """
        print(f"Running evaluation with smaller result set (k={num_results})...")
        
        precision_scores = []
        retrieval_times = []
        analysis_times = []
        total_times = []
        
        for _, case in self.test_cases.iterrows():
            description = case['description']
            ground_truth = set(case['ground_truth'].split(','))
            
            # Measure retrieval time
            retrieval_start = time.time()
            try:
                results = self.embedding_manager.search_similar_sections(description, k=num_results)
                retrieval_end = time.time()
                
                # Get retrieved sections
                retrieved_sections = []
                for doc, _ in results:
                    section = doc.metadata['section']
                    if isinstance(section, str) and section.startswith('Section '):
                        section_num = section.split(' ')[1]
                        retrieved_sections.append(section_num)
                    else:
                        retrieved_sections.append(str(section))
                
                # Calculate precision
                retrieved_set = set(retrieved_sections)
                relevant_retrieved = len(ground_truth.intersection(retrieved_set))
                precision = relevant_retrieved / len(retrieved_set) if retrieved_set else 0
            except Exception as e:
                print(f"Error retrieving results for '{description}': {e}")
                retrieval_end = time.time()
                precision = 0
                retrieved_sections = []
                results = []
            
            # Measure LLM analysis time if available
            try:
                formatted_results = [
                    {
                        'section': doc.metadata['section'], 
                        'content': doc.page_content,
                        'similarity': float(1 - score)  # Convert distance to similarity
                    } 
                    for doc, score in results
                ]
                
                analysis_start = time.time()
                _ = analyze_with_llm(description, formatted_results)
                analysis_end = time.time()
                analysis_time = analysis_end - analysis_start
            except Exception as e:
                print(f"LLM analysis unavailable or failed: {e}")
                analysis_time = 0
                
            # Record metrics
            retrieval_time = retrieval_end - retrieval_start
            total_time = retrieval_time + analysis_time
            
            precision_scores.append(precision)
            retrieval_times.append(retrieval_time)
            analysis_times.append(analysis_time)
            total_times.append(total_time)
        
        # Store results
        self.results['smaller_set'] = {
            'precision': np.mean(precision_scores),
            'retrieval_time': np.mean(retrieval_times),
            'analysis_time': np.mean(analysis_times),
            'total_time': np.mean(total_times)
        }
        
        print(f"Precision with smaller set: {self.results['smaller_set']['precision']:.4f}")
        print(f"Total time with smaller set: {self.results['smaller_set']['total_time']:.4f} seconds")
        
        return self.results['smaller_set']
    
    def run_all_ablation_studies(self, default_k=5, smaller_k=3):
        """
        Run all ablation studies
        
        Args:
            default_k: Default number of results to retrieve
            smaller_k: Number of results for smaller set test
        """
        print("Starting ablation studies...")
        
        # Create directory for plots
        plots_dir = "ablation_plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # Run all tests
        self.run_baseline(num_results=default_k)
        self.run_without_llm(num_results=default_k)
        self.run_smaller_results_set(num_results=smaller_k)
        
        return self.results
    
    def visualize_results(self):
        """Visualize the ablation study results"""
        if not self.results:
            print("No results available. Run ablation studies first.")
            return
            
        # Create directory for plots if it doesn't exist
        plots_dir = "ablation_plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # Prepare data for visualization
        configs = list(self.results.keys())
        precision_scores = [self.results[config]['precision'] for config in configs]
        
        # Plot precision comparison
        plt.figure(figsize=(10, 6))
        plt.bar(configs, precision_scores, color='skyblue')
        plt.title('Precision Comparison Across System Configurations')
        plt.ylabel('Average Precision')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'precision_comparison.png'))
        plt.close()
        
        # Plot time comparison
        retrieval_times = [self.results[config]['retrieval_time'] for config in configs]
        analysis_times = [self.results[config].get('analysis_time', 0) for config in configs]
        
        plt.figure(figsize=(10, 6))
        width = 0.35
        x = np.arange(len(configs))
        
        plt.bar(x, retrieval_times, width, label='Retrieval Time')
        plt.bar(x, analysis_times, width, bottom=retrieval_times, label='Analysis Time')
        
        plt.title('Response Time Breakdown Across System Configurations')
        plt.ylabel('Time (seconds)')
        plt.xticks(x, configs)
        plt.legend()
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'time_comparison.png'))
        plt.close()
        
        print(f"Visualizations saved to {plots_dir}/")
        
    def save_results(self, output_path='ablation_results.csv'):
        """
        Save the ablation study results to CSV
        
        Args:
            output_path: Path to save the results
        """
        if not self.results:
            print("No results to save. Run ablation studies first.")
            return
            
        # Convert results to DataFrame
        results_data = []
        for config, metrics in self.results.items():
            row = {'configuration': config}
            row.update(metrics)
            results_data.append(row)
            
        results_df = pd.DataFrame(results_data)
        
        # Save to CSV
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        
    def generate_report(self, output_path='ablation_report.md'):
        """
        Generate a markdown report of the ablation study results
        
        Args:
            output_path: Path to save the report
        """
        if not self.results:
            print("No results to report. Run ablation studies first.")
            return
            
        report = [
            "# Ablation Study Results",
            f"Date: {datetime.now().strftime('%Y-%m-%d')}",
            "",
            "## Overview",
            "",
            "This report presents the results of an ablation study performed on the Legal Assistance Platform.",
            "The study tested different system configurations to measure the impact of individual components on overall performance.",
            "",
            "## Configurations Tested",
            ""
        ]
        
        for config in self.results.keys():
            config_name = ' '.join(word.capitalize() for word in config.split('_'))
            report.append(f"- **{config_name}**")
        
        report.extend([
            "",
            "## Results Summary",
            "",
            "| Configuration | Precision | Retrieval Time (s) | Analysis Time (s) | Total Time (s) |",
            "|--------------|-----------|-------------------|------------------|---------------|"
        ])
        
        for config, metrics in self.results.items():
            precision = metrics['precision']
            retrieval_time = metrics['retrieval_time']
            analysis_time = metrics.get('analysis_time', 0)
            total_time = metrics['total_time']
            
            report.append(
                f"| {config} | {precision:.4f} | {retrieval_time:.4f} | {analysis_time:.4f} | {total_time:.4f} |"
            )
        
        report.extend([
            "",
            "## Key Findings",
            ""
        ])
        
        # Compare configurations
        baseline = self.results.get('baseline', {})
        if baseline:
            # Compare precision
            best_precision_config = max(self.results.items(), key=lambda x: x[1]['precision'])[0]
            best_precision = self.results[best_precision_config]['precision']
            
            # Compare speed
            fastest_config = min(self.results.items(), key=lambda x: x[1]['total_time'])[0]
            fastest_time = self.results[fastest_config]['total_time']
            
            report.extend([
                f"- The configuration with the highest precision was **{best_precision_config}** ({best_precision:.4f}).",
                f"- The fastest configuration was **{fastest_config}** ({fastest_time:.4f} seconds).",
                ""
            ])
            
            # Impact of LLM
            without_llm = self.results.get('without_llm', {})
            if without_llm:
                precision_change = without_llm['precision'] - baseline['precision']
                time_change = baseline['total_time'] - without_llm['total_time']
                
                report.extend([
                    "### Impact of LLM Analysis",
                    "",
                    f"- Removing LLM analysis changed precision by {precision_change:.4f} " + 
                    f"({'increased' if precision_change > 0 else 'decreased'}).",
                    f"- Removing LLM analysis saved {time_change:.4f} seconds per query.",
                    ""
                ])
            
            # Impact of result set size
            smaller_set = self.results.get('smaller_set', {})
            if smaller_set:
                precision_change = smaller_set['precision'] - baseline['precision']
                time_change = baseline['total_time'] - smaller_set['total_time']
                
                report.extend([
                    "### Impact of Result Set Size",
                    "",
                    f"- Using a smaller result set changed precision by {precision_change:.4f} " + 
                    f"({'increased' if precision_change > 0 else 'decreased'}).",
                    f"- Using a smaller result set saved {time_change:.4f} seconds per query.",
                    ""
                ])
        
        report.extend([
            "## Conclusion",
            "",
            "This ablation study provides insights into the contribution of each system component to overall performance.",
            "These findings can guide future optimization efforts and help balance precision and speed requirements.",
            ""
        ])
        
        # Write report to file
        with open(output_path, 'w') as file:
            file.write('\n'.join(report))
            
        print(f"Report saved to {output_path}")


if __name__ == "__main__":
    print("Initializing Ablation Study Manager...")
    # Use our prepared test data
    test_cases_path = "../../data/test_data/ablation_test_cases.csv"
    manager = AblationStudyManager(test_cases_path)
    
    print("\nRunning ablation studies...")
    manager.run_all_ablation_studies()
    
    print("\nVisualizing results...")
    manager.visualize_results()
    
    print("\nGenerating report...")
    manager.save_results()
    manager.generate_report()
    
    print("\nAblation study complete!")