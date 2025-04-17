"""
Master script to run all evaluations for the Legal Assistance Platform
"""
import os
import sys
import subprocess
import time
from datetime import datetime

def print_section(title):
    """Print a formatted section title"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def run_command(command, cwd=None):
    """Run a command and capture output"""
    try:
        print(f"Running: {command}")
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(f"Return code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False

def verify_test_data():
    """Verify test data exists"""
    print("Verifying test data...")
    
    test_files = [
        "data/test_data/test_cases.csv",
        "data/test_data/expert_annotations.csv",
        "data/test_data/ablation_test_cases.csv"
    ]
    
    all_exists = True
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path} exists")
        else:
            print(f"  ✗ {file_path} is missing")
            all_exists = False
            
    return all_exists

def run_technical_evaluations():
    """Run technical evaluations (accuracy and performance)"""
    print_section("Technical Evaluations")
    
    # Run accuracy metrics
    print("Running accuracy metrics evaluation...")
    success = run_command("python accuracy_metrics.py", cwd="evaluation/technical")
    
    if not success:
        print("Accuracy metrics evaluation failed. Check for errors above.")
    
    # Run performance metrics
    print("\nRunning performance metrics evaluation...")
    success = run_command("python performance_metrics.py", cwd="evaluation/technical")
    
    if not success:
        print("Performance metrics evaluation failed. Check for errors above.")

def run_ablation_study():
    """Run ablation studies"""
    print_section("Ablation Studies")
    
    success = run_command("python ablation_study.py", cwd="evaluation/comparative")
    
    if not success:
        print("Ablation study failed. Check for errors above.")

def generate_sample_survey():
    """Generate sample user survey form"""
    print_section("User Survey Generation")
    
    success = run_command("python survey_template.py", cwd="evaluation/user_studies")
    
    if not success:
        print("Survey generation failed. Check for errors above.")

def collect_results():
    """Collect and organize all results"""
    print_section("Collecting Results")
    
    # Create results directory if it doesn't exist
    results_dir = "evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Copy result files to results directory
    files_to_copy = [
        ("evaluation/technical/accuracy_results.csv", f"{results_dir}/accuracy_results.csv"),
        ("evaluation/technical/performance_results.csv", f"{results_dir}/performance_results.csv"),
        ("evaluation/technical/performance_results_concurrency.csv", f"{results_dir}/performance_concurrency.csv"),
        ("evaluation/technical/concurrency_performance.png", f"{results_dir}/concurrency_performance.png"),
        ("evaluation/comparative/ablation_results.csv", f"{results_dir}/ablation_results.csv"),
        ("evaluation/comparative/ablation_report.md", f"{results_dir}/ablation_report.md"),
        ("evaluation/comparative/ablation_plots/precision_comparison.png", f"{results_dir}/ablation_precision.png"),
        ("evaluation/comparative/ablation_plots/time_comparison.png", f"{results_dir}/ablation_time.png"),
        ("evaluation/user_studies/user_survey_form.html", f"{results_dir}/user_survey_form.html")
    ]
    
    for src, dst in files_to_copy:
        if os.path.exists(src):
            try:
                import shutil
                shutil.copy2(src, dst)
                print(f"Copied {src} to {dst}")
            except Exception as e:
                print(f"Error copying {src}: {e}")
        else:
            print(f"File not found: {src}")
    
    # Generate index.html to access all results
    index_html = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "    <meta charset='UTF-8'>",
        "    <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "    <title>Legal Assistance Platform - Evaluation Results</title>",
        "    <style>",
        "        body { font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }",
        "        h1, h2, h3 { color: #2c3e50; }",
        "        .section { margin-bottom: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 5px; }",
        "        img { max-width: 100%; height: auto; }",
        "        table { border-collapse: collapse; width: 100%; }",
        "        th, td { border: 1px solid #ddd; padding: 8px; }",
        "        th { background-color: #f2f2f2; }",
        "    </style>",
        "</head>",
        "<body>",
        "    <h1>Legal Assistance Platform - Evaluation Results</h1>",
        f"    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        "",
        "    <div class='section'>",
        "        <h2>Accuracy Metrics</h2>",
        "        <p><a href='accuracy_results.csv'>Download CSV</a></p>",
        "        <div id='accuracy-results'></div>",
        "    </div>",
        "",
        "    <div class='section'>",
        "        <h2>Performance Metrics</h2>",
        "        <p><a href='performance_results.csv'>Download CSV</a></p>",
        "        <div id='performance-results'></div>",
        "        <h3>Concurrency Testing</h3>",
        "        <p><a href='performance_concurrency.csv'>Download CSV</a></p>",
        "        <img src='concurrency_performance.png' alt='Concurrency Performance Chart'>",
        "    </div>",
        "",
        "    <div class='section'>",
        "        <h2>Ablation Studies</h2>",
        "        <p><a href='ablation_report.md'>View Full Report</a> | <a href='ablation_results.csv'>Download CSV</a></p>",
        "        <h3>Precision Comparison</h3>",
        "        <img src='ablation_precision.png' alt='Ablation Study Precision Comparison'>",
        "        <h3>Response Time Comparison</h3>",
        "        <img src='ablation_time.png' alt='Ablation Study Time Comparison'>",
        "    </div>",
        "",
        "    <div class='section'>",
        "        <h2>User Studies</h2>",
        "        <p><a href='user_survey_form.html'>View Survey Form</a></p>",
        "    </div>",
        "",
        "    <script>",
        "        // Function to load and display CSV data as tables",
        "        async function loadCSV(file, elementId) {",
        "            try {",
        "                const response = await fetch(file);",
        "                const text = await response.text();",
        "                const rows = text.split('\\n');",
        "                const headers = rows[0].split(',');",
        "                ",
        "                let tableHTML = '<table><thead><tr>';",
        "                headers.forEach(header => {",
        "                    tableHTML += `<th>${header}</th>`;",
        "                });",
        "                tableHTML += '</tr></thead><tbody>';",
        "                ",
        "                for(let i = 1; i < rows.length; i++) {",
        "                    if(rows[i].trim() === '') continue;",
        "                    const cells = rows[i].split(',');",
        "                    tableHTML += '<tr>';",
        "                    cells.forEach(cell => {",
        "                        tableHTML += `<td>${cell}</td>`;",
        "                    });",
        "                    tableHTML += '</tr>';",
        "                }",
        "                ",
        "                tableHTML += '</tbody></table>';",
        "                document.getElementById(elementId).innerHTML = tableHTML;",
        "            } catch(err) {",
        "                document.getElementById(elementId).innerHTML = `<p>Error loading data: ${err.message}</p>`;",
        "            }",
        "        }",
        "        ",
        "        // Load data when page loads",
        "        window.addEventListener('load', () => {",
        "            loadCSV('accuracy_results.csv', 'accuracy-results');",
        "            loadCSV('performance_results.csv', 'performance-results');",
        "        });",
        "    </script>",
        "</body>",
        "</html>"
    ]
    
    with open(f"{results_dir}/index.html", "w") as f:
        f.write("\n".join(index_html))
    
    print(f"\nResults collected and organized in {results_dir}/")
    print(f"View summary at {results_dir}/index.html")

if __name__ == "__main__":
    print_section("Legal Assistance Platform Evaluation")
    print("Starting evaluation process...")
    
    # Verify test data
    if not verify_test_data():
        print("Error: Some test data files are missing. Please ensure all test data is available.")
        sys.exit(1)
    
    # Run all evaluations
    run_technical_evaluations()
    run_ablation_study()
    generate_sample_survey()
    
    # Collect results
    collect_results()
    
    print_section("Evaluation Complete")
    print("All evaluations have been completed. Results are available in the 'evaluation_results' directory.")