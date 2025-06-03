#!/usr/bin/env python3
"""
LaTeX F1 Score Comparison Plot Generator

This script reads evaluation JSON files from multiple models and generates 
LaTeX code for comparing F1 scores across different entity classes.
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re


class F1ComparisonGenerator:
    def __init__(self):
        self.label_mappings = {
            # Common mappings between different label formats
            'facility': 'FAC',
            'location': 'LOC', 
            'work of art': 'WORK_OF_ART',
            'event': 'EVENT',
            'law': 'LAW',
            'national religious political': 'NORP',
            'product': 'PRODUCT',
            'injury or poisoning': 'INJURY_OR_POISONING',
            'virus': 'VIRUS',
            'clinical attribute': 'CLINICAL_ATTRIBUTE',
            'biomedical occupation or discipline': 'BIOMEDICAL_OCCUPATION_OR_DISCIPLINE',
            'organization': 'ORGANIZATION',
            'body system': 'BODY_SYSTEM',
            'food': 'FOOD',
            'body substance': 'BODY_SUBSTANCE',
            'bacterium': 'BACTERIUM',
            'professional or occupational group': 'PROFESSIONAL_OR_OCCUPATIONAL_GROUP',
            'judge': 'JUDGE',
            'org': 'ORG',
            'statute': 'STATUTE',
            'gpe': 'GPE',
            'geopolitical entity': 'GPE',
            'precedent': 'PRECEDENT',
            'case number': 'CASE_NUMBER',
            'witness': 'WITNESS',
        }
        
    def normalize_label(self, label: str) -> str:
        """Normalize label names for comparison"""
        label_lower = label.lower()
        return self.label_mappings.get(label_lower, label.upper())
    
    def load_evaluation_files(self, model_paths: List[str]) -> Dict[str, Dict[str, float]]:
        """Load evaluation files and extract F1 scores by label"""
        model_data = {}
        
        for model_path in model_paths:
            path_obj = Path(model_path)
            if not path_obj.exists():
                print(f"Warning: File {model_path} not found, skipping...")
                continue
                
            with open(path_obj, 'r') as f:
                data = json.load(f)
            
            # Extract model name from path
            model_name = self.extract_model_name(model_path)
            
            # Get F1 scores for each label
            if model_name not in model_data:
                model_data[model_name] = {}
                
            linker_data = data.get('linker', {})
            for label, metrics in linker_data.items():
                if model_name.lower() != 'zeroner' and label == 'organization' and 'legalner' in model_path:
                    label = 'org'
                    
                if isinstance(metrics, dict) and 'f1' in metrics:
                    normalized_label = self.normalize_label(label)
                    model_data[model_name][normalized_label] = metrics['f1']
        
        return model_data
    
    def extract_model_name(self, file_path: str) -> str:
        """Extract model name from file path"""
        path_parts = Path(file_path).parts
        
        if 'gliner_large' in file_path.lower():
            return 'GLiNER-L'
        elif 'gliner_small' in file_path.lower():
            return 'GLiNER-S'
        elif 'gliner_medium' in file_path.lower():
            return 'GLiNER-M'
        elif 'zeroner' in file_path.lower():
            return 'ZeroNER'
        else:
            # Extract from filename
            filename = Path(file_path).stem
            model_part = filename.split('_')[0]
            return model_part.replace('-', '\\_')
    
    def get_common_labels(self, model_data: Dict[str, Dict[str, float]]) -> List[str]:
        """Get labels that appear in all models"""
        if not model_data:
            return []
            
        model_names = list(model_data.keys())
        common_labels = set(model_data[model_names[0]].keys())
        #common_labels = {label.replace(" ", "_") for label in common_labels}
        #print(f"Initial common labels from {model_names[0]}: {common_labels}")
        #print(f"Initial common labels from {model_names[-1]}: {set(model_data[model_names[-1]].keys())}")
        for model_name in model_names[1:]:
            common_labels &= set(model_data[model_name].keys())
        print(f"Common labels found: {len(common_labels)}")
        print(f"Not common labels: {set(model_data[model_names[0]].keys()) - common_labels}")
        print(f"Not common labels: {set(model_data[model_names[-1]].keys()) - common_labels}")
        return sorted(list(common_labels))
    
    def format_label_for_latex(self, label: str) -> str:
        """Format label for LaTeX display"""
        # Convert common abbreviations to readable names
        label_display = {
            'FAC': 'FAC',
            'LOC': 'LOC',
            'WORK_OF_ART': 'Art',
            'EVENT': 'Event',
            'LAW': 'Law',
            'NORP': 'Norp',
            'PRODUCT': 'Product',
            "INJURY_OR_POISONING": "Injury",
            "VIRUS": "Virus",
            "CLINICAL_ATTRIBUTE": "Clinic Attr",
            "BIOMEDICAL_OCCUPATION_OR_DISCIPLINE": "Bio Occupation",
            "ORGANIZATION": "ORG",
            "BODY_SYSTEM": "Body System",
            "FOOD": "Food",
            "BODY_SUBSTANCE": "Substance",
            "BACTERIUM": "Bacterium",
            "PROFESSIONAL_OR_OCCUPATIONAL_GROUP": "Occupation",
            "JUDGE": "Judge",
            "ORG": "Legal ORG",
            "STATUTE": "Statute",
            "GPE": "GPE",
            "PRECEDENT": "Precedent",
            "CASE_NUMBER": "Case Number",
            "WITNESS": "Witness"
           
        }
        return label_display.get(label, label.title())
    
    def calculate_y_positions(self, model_data: Dict[str, Dict[str, float]], 
                            common_labels: List[str]) -> List[float]:
        """Calculate Y positions for difference annotations"""
        positions = []
        x_positions = [1.1, 4.3, 7.6]  # Base positions for up to 3 labels
        
        for i, label in enumerate(common_labels):
            max_f1 = max(model_data[model][label] for model in model_data.keys())
            # Position slightly above the highest bar
            y_pos = (max_f1 + 0.05) * 7 / 0.65  # Scale to plot coordinates
            positions.append(y_pos)
        
        return positions
    
    def generate_latex_code(self, model_data: Dict[str, Dict[str, float]], 
                          output_file: Optional[str] = None,
                          figure_caption: str = None) -> str:
        """Generate LaTeX code for the comparison plot"""
        
        common_labels = self.get_common_labels(model_data)
        if len(common_labels) == 0:
            raise ValueError("No common labels found between models")
        
        model_names = list(model_data.keys())
        if len(model_names) > 4:
            raise ValueError("Maximum 4 models supported")
        if len(model_names) < 2:
            raise ValueError("At least 2 models required for comparison")
        
        # Format labels for display
        display_labels = [self.format_label_for_latex(label) for label in common_labels]
        
        # Determine y_max based on all models
        max_f1 = 0
        for label in common_labels:
            for model in model_names:
                max_f1 = max(max_f1, model_data[model][label])
        
        y_max = min(1.0, max_f1 + 0.20)  # Add padding for F1 score labels, cap at 1.0
        
        # Generate coordinates for each model
        model_coords = []
        colors = ['blue!60', 'red!60', 'green!60', 'orange!60']
        border_colors = ['blue!80', 'red!80', 'green!80', 'orange!80']
        
        for model_idx, model_name in enumerate(model_names):
            coords = []
            for label in common_labels:
                display_label = self.format_label_for_latex(label)
                coords.append(f"        ({display_label}, {model_data[model_name][label]:.2f})")
            model_coords.append((coords, colors[model_idx], border_colors[model_idx]))
        
        # Calculate annotation positions for F1 scores
        num_labels = len(common_labels)
        num_models = len(model_names)
        
        # Calculate base x positions for each entity class
        if num_labels == 1:
            base_x_positions = [5.0]
        elif num_labels == 2:
            base_x_positions = [2.5, 7.5]
        elif num_labels == 3:
            base_x_positions = [1.7, 5.0, 8.3]
        else:
            # For 4+ labels, distribute evenly with some padding
            plot_width = 10.0  # from axis width
            spacing = plot_width / (num_labels + 1)
            base_x_positions = [spacing * (i + 1) for i in range(num_labels)]
        
        # Generate F1 score annotations for each model and label
        annotations = []
        bar_width = 0.4  # from LaTeX code
        total_bar_width = bar_width * num_models
        
        for label_idx, label in enumerate(common_labels):
            base_x = base_x_positions[label_idx]
            
            # Calculate individual bar positions within each group
            for model_idx, model_name in enumerate(model_names):
                # Offset from center of group
                if num_models == 2:
                    bar_offset = (model_idx - 0.5) * bar_width
                elif num_models == 3:
                    bar_offset = (model_idx - 1) * bar_width
                elif num_models == 4:
                    bar_offset = (model_idx - 1.5) * bar_width
                else:
                    bar_offset = 0
                
                x_pos = base_x + bar_offset
                f1_score = model_data[model_name][label]
                y_pos = (f1_score + 0.03) * 7 / y_max  # Slightly above the bar
                
                annotations.append(f"    \\node[anchor=south, font=\\tiny] at ({x_pos:.1f}, {y_pos:.1f}) {{{f1_score:.2f}}};")
        
        # Default caption
        if not figure_caption:
            model_list = ", ".join(model_names[:-1]) + f" and {model_names[-1]}" if len(model_names) > 2 else f"{model_names[0]} and {model_names[1]}"
            figure_caption = f"F1 Score comparison between {model_list} across different entity classes. Values above bars show the F1 scores for each model."
        
        # Generate LaTeX code - using regular string concatenation to avoid f-string backslash issues
        newline = chr(10)
        escaped_model_names = [name.replace('_', '\\_') for name in model_names]
        
        # Generate plot commands for each model
        plot_commands = []
        for model_idx, (coords, fill_color, border_color) in enumerate(model_coords):
            plot_commands.append(f"""    % {model_names[model_idx]} data
    \\addplot[
        fill={fill_color},
        draw={border_color},
    ] coordinates {{
{newline.join(coords)}
    }};""")
        
        # Adjust bar width based on number of models
        if num_models <= 2:
            bar_width_str = "0.3cm"
        elif num_models == 3:
            bar_width_str = "0.2cm"
        else:
            bar_width_str = "0.15cm"
        
        latex_code = f"""\\begin{{figure*}}[htbp]
    \\centering
    \\resizebox{{\\linewidth}}{{!}}{{%
    \\begin{{tikzpicture}}
    \\begin{{axis}}[
        width=12cm,
        height=6cm,
        ybar,
        bar width={bar_width_str},
        ylabel={{F1 Score}},
        xlabel={{Entity Classes}},
        symbolic x coords={{{', '.join(display_labels)}}},
        xtick=data,
        x tick label style={{
            font=\\tiny, 
            rotate=45,
            anchor=east,  
            align=right
        }},
        ymin=0,
        ymax={y_max:.2f},
        legend style={{at={{(0.5,1.05)}},
            anchor=south,
            legend columns=-1,
            font=\\tiny,
            /tikz/every even column/.append style={{column sep=0.5cm}}
        }},
        legend cell align={{left}}
        grid=major,
        grid style={{opacity=0.3}},
        enlarge x limits=0.1,
    ]
    
{newline.join(plot_commands)}
    
    \\legend{{{', '.join(escaped_model_names)}}}
    \\end{{axis}}
    
    % Add F1 score annotations above each bar
{newline.join(annotations)}
    
    \\end{{tikzpicture}}
    }}
    \\caption{{{figure_caption}}}
    \\label{{fig:f1_comparison}}
\\end{{figure*}}"""
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(latex_code)
            print(f"LaTeX code written to {output_file}")
        
        return latex_code


def main():
    parser = argparse.ArgumentParser(description='Generate LaTeX F1 comparison plots')
    parser.add_argument('--model1_files', nargs='+', required=True,
                       help='JSON files for model 1')
    parser.add_argument('--model2_files', nargs='+', required=True, 
                       help='JSON files for model 2')
    parser.add_argument('--model3_files', nargs='+', required=False,
                       help='JSON files for model 3 (optional)')
    parser.add_argument('--model4_files', nargs='+', required=False,
                       help='JSON files for model 4 (optional)')
    parser.add_argument('--output', '-o', help='Output LaTeX file')
    parser.add_argument('--caption', help='Custom figure caption')
    
    args = parser.parse_args()
    
    generator = F1ComparisonGenerator()
    
    # Combine all files for processing
    all_files = args.model1_files + args.model2_files
    if args.model3_files:
        all_files += args.model3_files
    if args.model4_files:
        all_files += args.model4_files
    
    try:
        # Load data from all files
        model_data = generator.load_evaluation_files(all_files)
        
        if len(model_data) == 0:
            print("Error: No valid evaluation files found")
            return
        
        # Generate LaTeX code
        latex_code = generator.generate_latex_code(
            model_data, 
            args.output,
            args.caption
        )
        
        if not args.output:
            print("Generated LaTeX code:")
            print("=" * 50)
            print(latex_code)
        
        # Print summary
        print(f"\\nProcessed {len(model_data)} models:")
        for model_name, labels in model_data.items():
            print(f"  - {model_name}: {len(labels)} labels")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()