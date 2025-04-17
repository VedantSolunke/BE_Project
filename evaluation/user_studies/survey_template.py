"""
User Study Survey Template for Legal Assistance Platform
"""
import sys
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime

class UserSurveyManager:
    def __init__(self):
        """Initialize the user survey manager"""
        self.survey_questions = {
            'demographics': [
                {
                    'id': 'role',
                    'text': 'What is your professional role?',
                    'type': 'selection',
                    'options': ['Law Student', 'Legal Practitioner', 'Judge', 'Law Professor', 'Legal Researcher', 'Other']
                },
                {
                    'id': 'experience',
                    'text': 'Years of experience in the legal field:',
                    'type': 'selection',
                    'options': ['0-2 years', '3-5 years', '6-10 years', '11-20 years', 'More than 20 years']
                },
                {
                    'id': 'tech_comfort',
                    'text': 'How would you rate your comfort level with legal technology tools?',
                    'type': 'rating',
                    'scale': 5,
                    'labels': ['Not comfortable', 'Somewhat comfortable', 'Neutral', 'Comfortable', 'Very comfortable']
                }
            ],
            'system_usability': [
                {
                    'id': 'ease_of_use',
                    'text': 'How easy was it to use the platform?',
                    'type': 'rating',
                    'scale': 5,
                    'labels': ['Very difficult', 'Difficult', 'Neutral', 'Easy', 'Very easy']
                },
                {
                    'id': 'interface_clarity',
                    'text': 'How clear and intuitive was the user interface?',
                    'type': 'rating',
                    'scale': 5,
                    'labels': ['Very confusing', 'Confusing', 'Neutral', 'Clear', 'Very clear']
                },
                {
                    'id': 'response_time',
                    'text': 'How would you rate the system\'s response time?',
                    'type': 'rating',
                    'scale': 5,
                    'labels': ['Very slow', 'Slow', 'Acceptable', 'Fast', 'Very fast']
                }
            ],
            'result_quality': [
                {
                    'id': 'relevance',
                    'text': 'How relevant were the IPC sections returned for your test cases?',
                    'type': 'rating',
                    'scale': 5,
                    'labels': ['Not relevant', 'Somewhat relevant', 'Moderately relevant', 'Relevant', 'Highly relevant']
                },
                {
                    'id': 'completeness',
                    'text': 'How complete was the set of returned IPC sections (were any important sections missing)?',
                    'type': 'rating',
                    'scale': 5,
                    'labels': ['Very incomplete', 'Missing sections', 'Moderately complete', 'Mostly complete', 'Very complete']
                },
                {
                    'id': 'accuracy',
                    'text': 'How accurate was the LLM analysis of the case?',
                    'type': 'rating',
                    'scale': 5,
                    'labels': ['Not accurate', 'Somewhat accurate', 'Moderately accurate', 'Accurate', 'Highly accurate']
                },
                {
                    'id': 'ranking_quality',
                    'text': 'How well were the results ranked by relevance?',
                    'type': 'rating',
                    'scale': 5,
                    'labels': ['Poor ranking', 'Fair ranking', 'Acceptable ranking', 'Good ranking', 'Excellent ranking']
                }
            ],
            'comparative': [
                {
                    'id': 'compared_to_manual',
                    'text': 'Compared to manual legal research, this system is:',
                    'type': 'selection',
                    'options': ['Much worse', 'Worse', 'About the same', 'Better', 'Much better']
                },
                {
                    'id': 'compared_to_existing',
                    'text': 'Compared to existing legal search tools you\'ve used, this system is:',
                    'type': 'selection',
                    'options': ['Much worse', 'Worse', 'About the same', 'Better', 'Much better', 'N/A']
                }
            ],
            'task_completion': [
                {
                    'id': 'task_completion_time',
                    'text': 'Approximately how long did it take you to complete the assigned task?',
                    'type': 'selection',
                    'options': ['Less than 2 minutes', '2-5 minutes', '5-10 minutes', '10-15 minutes', 'More than 15 minutes']
                },
                {
                    'id': 'task_difficulty',
                    'text': 'How difficult was it to complete the assigned task using this system?',
                    'type': 'rating',
                    'scale': 5,
                    'labels': ['Very difficult', 'Difficult', 'Moderate', 'Easy', 'Very easy']
                }
            ],
            'open_ended': [
                {
                    'id': 'strengths',
                    'text': 'What were the main strengths of the system?',
                    'type': 'text'
                },
                {
                    'id': 'weaknesses',
                    'text': 'What were the main weaknesses or areas for improvement?',
                    'type': 'text'
                },
                {
                    'id': 'suggestions',
                    'text': 'Do you have any specific suggestions for improving the system?',
                    'type': 'text'
                }
            ]
        }
        
        self.task_scenarios = [
            {
                'id': 'scenario_1',
                'title': 'Theft Case',
                'description': 'A person reports that their laptop was stolen from their hotel room while they were out for dinner. There was no sign of forced entry, but they are certain they locked the door. The hotel staff had access to master keys.',
                'expected_sections': ['Section 378', 'Section 380']
            },
            {
                'id': 'scenario_2',
                'title': 'Assault Case',
                'description': 'During an argument, person A pushed person B, causing them to fall and suffer a fractured wrist. Person A claims they did not intend to cause serious injury.',
                'expected_sections': ['Section 351', 'Section 323', 'Section 325']
            },
            {
                'id': 'scenario_3',
                'title': 'Digital Fraud Case',
                'description': 'A person received an email claiming to be from their bank, asking them to update their account details. After clicking the link and entering their information, they discovered unauthorized transactions in their account totaling Rs. 50,000.',
                'expected_sections': ['Section 420', 'Section 66C', 'Section 66D']
            }
        ]
        
        self.results_template = {
            'participant_id': '',
            'timestamp': '',
            'demographics': {},
            'system_usability': {},
            'result_quality': {},
            'comparative': {},
            'task_completion': {},
            'open_ended': {},
            'tasks': {}
        }
        
    def generate_survey_form(self, output_path='user_survey_form.html'):
        """
        Generate an HTML survey form based on the question templates
        
        Args:
            output_path: Path to save the generated HTML form
        """
        html_content = [
            '<!DOCTYPE html>',
            '<html lang="en">',
            '<head>',
            '    <meta charset="UTF-8">',
            '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            '    <title>Legal Assistance Platform Evaluation Survey</title>',
            '    <style>',
            '        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }',
            '        h1, h2, h3 { color: #2c3e50; }',
            '        .section { margin-bottom: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 5px; }',
            '        .question { margin-bottom: 20px; }',
            '        label { display: block; margin-bottom: 5px; font-weight: bold; }',
            '        input[type="text"], textarea { width: 100%; padding: 8px; }',
            '        textarea { height: 100px; }',
            '        .rating { display: flex; flex-direction: row; margin-top: 5px; }',
            '        .rating label { margin-right: 10px; font-weight: normal; }',
            '        button { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; }',
            '    </style>',
            '</head>',
            '<body>',
            '    <h1>Legal Assistance Platform Evaluation Survey</h1>',
            '    <p>Thank you for participating in this evaluation of our Legal Assistance Platform. Your feedback is valuable for improving the system.</p>',
            '    <form id="surveyForm">'
        ]
        
        # Add participant ID field
        html_content.extend([
            '    <div class="section">',
            '        <h2>Participant Information</h2>',
            '        <div class="question">',
            '            <label for="participant_id">Participant ID:</label>',
            '            <input type="text" id="participant_id" name="participant_id" required>',
            '        </div>',
            '    </div>'
        ])
        
        # Generate questions for each section
        for section_name, questions in self.survey_questions.items():
            section_title = ' '.join(word.capitalize() for word in section_name.split('_'))
            
            html_content.extend([
                f'    <div class="section">',
                f'        <h2>{section_title}</h2>'
            ])
            
            for question in questions:
                q_id = question['id']
                q_text = question['text']
                q_type = question['type']
                
                html_content.extend([
                    f'        <div class="question">',
                    f'            <label for="{q_id}">{q_text}</label>'
                ])
                
                if q_type == 'text':
                    html_content.append(f'            <textarea id="{q_id}" name="{q_id}"></textarea>')
                elif q_type == 'rating':
                    html_content.append(f'            <div class="rating">')
                    for i in range(1, question['scale'] + 1):
                        label = question['labels'][i-1] if i <= len(question['labels']) else str(i)
                        html_content.extend([
                            f'                <label>',
                            f'                    <input type="radio" name="{q_id}" value="{i}">',
                            f'                    {i} - {label}',
                            f'                </label>'
                        ])
                    html_content.append(f'            </div>')
                elif q_type == 'selection':
                    html_content.append(f'            <select id="{q_id}" name="{q_id}">')
                    html_content.append(f'                <option value="">-- Please select --</option>')
                    for option in question['options']:
                        html_content.append(f'                <option value="{option}">{option}</option>')
                    html_content.append(f'            </select>')
                
                html_content.append(f'        </div>')
            
            html_content.append(f'    </div>')
        
        # Add task scenario evaluations
        html_content.extend([
            '    <div class="section">',
            '        <h2>Task Scenarios</h2>',
            '        <p>For each scenario, please use the Legal Assistance Platform to find relevant IPC sections, then answer the following questions:</p>'
        ])
        
        for scenario in self.task_scenarios:
            scenario_id = scenario['id']
            scenario_title = scenario['title']
            scenario_desc = scenario['description']
            
            html_content.extend([
                f'        <div class="scenario">',
                f'            <h3>{scenario_title}</h3>',
                f'            <p><strong>Scenario:</strong> {scenario_desc}</p>',
                f'            <div class="question">',
                f'                <label for="{scenario_id}_relevance">How relevant were the returned IPC sections for this case?</label>',
                f'                <div class="rating">'
            ])
            
            for i in range(1, 6):
                labels = ['Not relevant', 'Somewhat relevant', 'Moderately relevant', 'Relevant', 'Highly relevant']
                label = labels[i-1] if i <= len(labels) else str(i)
                html_content.extend([
                    f'                    <label>',
                    f'                        <input type="radio" name="{scenario_id}_relevance" value="{i}">',
                    f'                        {i} - {label}',
                    f'                    </label>'
                ])
            
            html_content.extend([
                f'                </div>',
                f'            </div>',
                f'            <div class="question">',
                f'                <label for="{scenario_id}_sections">What IPC sections were returned? (Enter section numbers separated by commas)</label>',
                f'                <input type="text" id="{scenario_id}_sections" name="{scenario_id}_sections">',
                f'            </div>',
                f'            <div class="question">',
                f'                <label for="{scenario_id}_missing">Were any important sections missing? If so, which ones?</label>',
                f'                <input type="text" id="{scenario_id}_missing" name="{scenario_id}_missing">',
                f'            </div>',
                f'            <div class="question">',
                f'                <label for="{scenario_id}_comments">Any additional comments on this scenario?</label>',
                f'                <textarea id="{scenario_id}_comments" name="{scenario_id}_comments"></textarea>',
                f'            </div>',
                f'        </div>'
            ])
        
        html_content.extend([
            '    </div>',
            '    <button type="submit">Submit Survey</button>',
            '    </form>',
            '    <script>',
            '        document.getElementById("surveyForm").addEventListener("submit", function(event) {',
            '            event.preventDefault();',
            '            alert("Thank you for completing the survey! In a real implementation, this data would be saved.");',
            '            // In a real implementation, data would be sent to a server or saved locally',
            '        });',
            '    </script>',
            '</body>',
            '</html>'
        ])
        
        with open(output_path, 'w') as file:
            file.write('\n'.join(html_content))
            
        print(f"Survey form generated and saved to {output_path}")
        
    def analyze_survey_results(self, results_file, output_dir='survey_results'):
        """
        Analyze survey results from a CSV file
        
        Args:
            results_file: Path to CSV file with survey results
            output_dir: Directory to save analysis results
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load results data
        try:
            results = pd.read_csv(results_file)
        except FileNotFoundError:
            print(f"Error: Results file '{results_file}' not found.")
            return
        except Exception as e:
            print(f"Error loading results file: {e}")
            return
            
        # Basic statistics
        num_participants = len(results)
        print(f"Number of participants: {num_participants}")
        
        # Demographics analysis
        if 'role' in results.columns:
            role_counts = results['role'].value_counts()
            
            plt.figure(figsize=(10, 6))
            role_counts.plot(kind='bar')
            plt.title('Participant Roles')
            plt.xlabel('Role')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'participant_roles.png'))
            
        # System usability analysis
        usability_columns = [col for col in results.columns if any(col.startswith(prefix) for prefix in ['ease_of_use', 'interface_clarity', 'response_time'])]
        
        if usability_columns:
            usability_scores = results[usability_columns].mean()
            
            plt.figure(figsize=(10, 6))
            usability_scores.plot(kind='bar')
            plt.title('System Usability Scores')
            plt.xlabel('Metric')
            plt.ylabel('Average Score (1-5)')
            plt.ylim(0, 5)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'usability_scores.png'))
            
        # Result quality analysis
        quality_columns = [col for col in results.columns if any(col.startswith(prefix) for prefix in ['relevance', 'completeness', 'accuracy', 'ranking_quality'])]
        
        if quality_columns:
            quality_scores = results[quality_columns].mean()
            
            plt.figure(figsize=(10, 6))
            quality_scores.plot(kind='bar')
            plt.title('Result Quality Scores')
            plt.xlabel('Metric')
            plt.ylabel('Average Score (1-5)')
            plt.ylim(0, 5)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'quality_scores.png'))
            
        # Generate summary report
        summary_report = [
            "# User Survey Analysis Summary",
            f"Date: {datetime.now().strftime('%Y-%m-%d')}",
            f"Number of participants: {num_participants}",
            "",
            "## Key Findings"
        ]
        
        # Add mean scores for different categories
        categories = {
            'System Usability': usability_columns,
            'Result Quality': quality_columns
        }
        
        for category, cols in categories.items():
            if cols:
                mean_score = results[cols].mean().mean()
                summary_report.append(f"- Average {category} score: {mean_score:.2f}/5.0")
        
        # Add task completion information
        if 'task_completion_time' in results.columns:
            time_counts = results['task_completion_time'].value_counts()
            most_common_time = time_counts.idxmax()
            summary_report.append(f"- Most common task completion time: {most_common_time}")
        
        # Add comparative information
        if 'compared_to_manual' in results.columns:
            better_count = results[results['compared_to_manual'].isin(['Better', 'Much better'])].shape[0]
            better_percentage = (better_count / num_participants) * 100
            summary_report.append(f"- {better_percentage:.1f}% found the system better than manual research")
        
        # Write summary report
        with open(os.path.join(output_dir, 'summary_report.md'), 'w') as file:
            file.write('\n'.join(summary_report))
            
        print(f"Analysis complete. Results saved to {output_dir}/")
        
        
if __name__ == "__main__":
    print("Initializing User Survey Manager...")
    manager = UserSurveyManager()
    
    # Generate survey form
    print("\nGenerating survey form...")
    manager.generate_survey_form()
    
    print("\nExample usage for analyzing results:")
    print("- Collect survey responses in CSV format")
    print("- Run: manager.analyze_survey_results('survey_responses.csv')")
    print("\nNote: This is a template. Actual data collection implementation would depend on your deployment setup.")