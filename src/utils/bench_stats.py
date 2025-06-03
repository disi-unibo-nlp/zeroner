import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any
import numpy as np
from datasets import load_dataset, Dataset
import pandas as pd

class NERDatasetAnalyzer:
    def __init__(self):
        # Store current dataset for tag scheme detection
        self.current_dataset = None
        
        # Predefined tag mappings for known datasets
        self.dataset_tag_mappings = {}
        
    def load_hf_dataset(self, dataset_name: str, split: str = None) -> Dataset:
        """Load dataset from Hugging Face"""

        if "ontonotes" in dataset_name.lower():
            from zshot.evaluation.dataset import load_ontonotes_zs
            dataset = load_ontonotes_zs()
            self.current_dataset = dataset
            return dataset
        else:
            try:
                if split:
                    dataset = load_dataset(dataset_name, split=split)
                else:
                    dataset = load_dataset(dataset_name)
                
                # Store dataset reference for tag scheme detection
                self.current_dataset = dataset
                return dataset
            except Exception as e:
                print(f"Error loading {dataset_name}: {e}")
                return None
    
    def get_tag_scheme(self, ner_tags: List[int], dataset_name: str = None) -> Dict[int, str]:
        """Automatically detect or retrieve tag scheme for the dataset"""
        unique_tags = set(ner_tags)

        """
        # Try to get feature info from dataset if available
        if hasattr(self, 'current_dataset') and self.current_dataset:
            try:
                # Get first split to check features
                first_split = list(self.current_dataset.keys())[0]
                split_data = self.current_dataset[first_split]
                
                # Check if dataset has tag names
                if 'ner_tags' in split_data.features:
                    feature = split_data.features['ner_tags']
                    if hasattr(feature, 'feature') and hasattr(feature.feature, 'names'):
                        tag_names = feature.feature.names
                        # Return mapping for all available tags
                        return {i: tag_names[i] for i in range(len(tag_names))}
            except Exception as e:
                print(f"Error accessing dataset features: {e}")
                pass
        
        # Use predefined mapping if available
        if dataset_name and dataset_name in self.dataset_tag_mappings:
            return self.dataset_tag_mappings[dataset_name]
        """

        # Generic mapping for unknown datasets
        tag_mapping = {}
        for tag in unique_tags:
            if tag == 0:
                tag_mapping[tag] = 'O'  # Always map 0 to 'O' (outside)
            else:
                tag_mapping[tag] = f'TAG_{tag}'
        
        return tag_mapping
    
    def debug_tag_scheme(self, dataset_name: str, sample_size: int = 100):
        """Debug function to print tag scheme information"""
        print(f"\nDEBUG: Tag scheme for {dataset_name}")
        print("="*50)
        
        dataset = self.load_hf_dataset(dataset_name)
        if not dataset:
            return
        
        # Get first split
        first_split = list(dataset.keys())[0]
        split_data = dataset[first_split]
        
        # Collect unique tags from sample
        all_tags = set()
        sample_examples = []
        
        for i, example in enumerate(split_data):
            if i >= sample_size:
                break
            all_tags.update(example['ner_tags'])
            sample_examples.append(example)
        
        print(f"Dataset features: {list(dataset[first_split].features.keys())}")
        
        # Try to get actual tag names from the dataset
        try:
            if 'ner_tags' in dataset[first_split].features:
                feature = dataset[first_split].features['ner_tags']
                print(f"NER tags feature type: {type(feature)}")
                if hasattr(feature, 'feature'):
                    print(f"Feature.feature type: {type(feature.feature)}")
                    if hasattr(feature.feature, 'names'):
                        actual_tag_names = feature.feature.names
                        print(f"Actual tag names from dataset: {actual_tag_names}")
                        # Use actual tag names
                        tag_mapping = {i: actual_tag_names[i] for i in range(len(actual_tag_names)) if i in all_tags}
                    else:
                        print("No 'names' attribute found in feature.feature")
                        tag_mapping = self.get_tag_scheme(list(all_tags), dataset_name)
                else:
                    print("No 'feature' attribute found in feature")
                    tag_mapping = self.get_tag_scheme(list(all_tags), dataset_name)
            else:
                print("No 'ner_tags' feature found")
                tag_mapping = self.get_tag_scheme(list(all_tags), dataset_name)
        except Exception as e:
            print(f"Error getting tag names: {e}")
            tag_mapping = self.get_tag_scheme(list(all_tags), dataset_name)
        
        # Print tag mapping
        print("Tag mapping:")
        for tag_id in sorted(all_tags):
            tag_name = tag_mapping.get(tag_id, f"UNKNOWN_{tag_id}")
            print(f"  {tag_id}: {tag_name}")
        
        # Print a few example sentences with their tags
        print(f"\nExample sentences (first 3):")
        for i, example in enumerate(sample_examples[:3]):
            tokens = example['tokens']
            ner_tags = example['ner_tags']
            print(f"\nSentence {i+1}:")
            print(f"Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
            print(f"Tags: {[tag_mapping.get(t, f'UNK_{t}') for t in ner_tags[:10]]}{'...' if len(ner_tags) > 10 else ''}")
            
            # Extract entities for this example
            entities = self.extract_entities(tokens, ner_tags, dataset_name)
            print(f"Entities found: {len([el for el in entities if el[1].startswith('B-')])}")
            for entity_text, entity_type, start, end in entities[:5]:
                print(f"  '{entity_text}' ({entity_type}) - tokens {start}-{end}")
        
        print("="*50)
    
    def extract_entities(self, tokens: List[str], ner_tags: List[int], dataset_name: str = None) -> List[Tuple[str, str, int, int]]:
        """Extract entities from BIO tagged sequence"""
        entities = []
        current_entity = None
        current_tokens = []
        start_idx = 0
        
        # Get appropriate tag mapping
        tag_mapping = self.get_tag_scheme(ner_tags, dataset_name)
        
        for i, (token, tag) in enumerate(zip(tokens, ner_tags)):
            tag_str = tag_mapping.get(tag, f"TAG_{tag}")
            
            # Skip 'O' tags (outside entity) - these should not be counted as entities
            if tag_str == 'O' or tag_str.startswith('TAG_O') or tag == 0:
                # Save current entity if exists
                if current_entity:
                    entity_text = ' '.join(current_tokens)
                    entities.append((entity_text, current_entity, start_idx, start_idx + len(current_tokens) - 1))
                    current_entity = None
                    current_tokens = []
                continue
            
            if tag_str.startswith('TAG_B-'):
                # Save previous entity if exists
                if current_entity:
                    entity_text = ' '.join(current_tokens)
                    entities.append((entity_text, current_entity, start_idx, start_idx + len(current_tokens) - 1))
                
                # Start new entity
                current_entity = tag_str[4:]  # Remove 'TAG_B-'
                current_tokens = [token]
                start_idx = i
                
            elif tag_str.startswith('TAG_I-'):
                # Continue current entity
                if current_entity and tag_str[4:] == current_entity:
                    current_tokens.append(token)
                else:
                    # Either no current entity or entity type changed
                    if current_entity:
                        # Save previous entity
                        entity_text = ' '.join(current_tokens)
                        entities.append((entity_text, current_entity, start_idx, start_idx + len(current_tokens) - 1))
                    
                    # Start new entity from I- tag (handle cases where B- is missing)
                    current_entity = tag_str[4:]
                    current_tokens = [token]
                    start_idx = i
                    
            elif tag_str.startswith('TAG_') and not tag_str.startswith('TAG_O') and tag != 0:
                # Handle generic non-BIO entity tags, but skip O tags
                if current_entity:
                    # Save previous entity
                    entity_text = ' '.join(current_tokens)
                    entities.append((entity_text, current_entity, start_idx, start_idx + len(current_tokens) - 1))
                
                # For generic tags, treat each token as a separate entity
                current_entity = tag_str
                current_tokens = [token]
                start_idx = i
                # Immediately close this single-token entity
                entity_text = ' '.join(current_tokens)
                entities.append((entity_text, current_entity, start_idx, start_idx))
                current_entity = None
                current_tokens = []
        
        # Don't forget the last entity
        if current_entity:
            entity_text = ' '.join(current_tokens)
            entities.append((entity_text, current_entity, start_idx, start_idx + len(current_tokens) - 1))
            
        return entities
    
    def find_consecutive_same_class_entities(self, ner_tags: List[int], dataset_name: str = None) -> int:
        """Count consecutive B-tags of the same entity class (BIO tagging violations)"""
        consecutive_count = 0
        
        # Get appropriate tag mapping
        tag_mapping = self.get_tag_scheme(ner_tags, dataset_name)
        
        for i in range(len(ner_tags) - 1):
            current_tag = tag_mapping.get(ner_tags[i], f"TAG_{ner_tags[i]}")
            next_tag = tag_mapping.get(ner_tags[i + 1], f"TAG_{ner_tags[i + 1]}")
            
            # Check if both are B-tags of the same class
            if (current_tag.startswith('TAG_B-') and next_tag.startswith('TAG_B-') and
                current_tag[4:] == next_tag[4:]):  # Same entity class
                consecutive_count += 1
                #print("BIO violation found:", current_tag, next_tag)
                #print("Ner tags:", ner_tags)
                
        return consecutive_count
    
    def find_compound_entities(self, entities: List[Tuple[str, str, int, int]]) -> int:
        """Count compound entities (entities with multiple tokens)"""
        compound_count = 0
        for entity_text, entity_type, start_idx, end_idx in entities:
            # An entity is compound if it spans multiple tokens (end_idx > start_idx)
            # OR if the entity text contains multiple words
            if end_idx > start_idx or len(entity_text.split()) > 1:
                compound_count += 1
        return compound_count
    
    def analyze_split(self, data: List[Dict], dataset_name: str = None) -> Dict[str, Any]:
        """Analyze a single dataset split"""
        stats = {
            'num_sentences': 0,
            'num_words': 0,
            'num_entities': 0,
            'num_entity_words': 0,
            'num_compound_entities': 0,
            'num_consecutive_same_class': 0,
            'entity_types': set(),
            #'entity_lengths': [],
            'sentence_lengths': [],
            'entity_type_counts': Counter(),
            'entities_per_sentence': []
        }
        
        for example in data:
            tokens = example['tokens']
            ner_tags = example['ner_tags']
            
            # Basic counts
            stats['num_sentences'] += 1
            stats['num_words'] += len(tokens)
            stats['sentence_lengths'].append(len(tokens))
            
            # Extract entities
            entities = self.extract_entities(tokens, ner_tags, dataset_name)
            
            stats['num_entities'] += len([el for el in entities if el[1].startswith("B-")])
            stats['entities_per_sentence'].append(len([el for el in entities if el[1].startswith("B-")]))
            stats['num_entity_words'] += len(entities)
            
            # Analyze entities
            for entity_text, entity_type, start_idx, end_idx in entities:
                if entity_type.startswith("B-"):
                    stats['entity_types'].add(entity_type)
                
                stats['entity_type_counts'][entity_type] += 1
                #stats['entity_lengths'].append(len(entity_text.split()))
            
            # Count compound entities
            stats['num_compound_entities'] += self.find_compound_entities(entities)
            
            # Count consecutive same-class entities (BIO violations)
            stats['num_consecutive_same_class'] += self.find_consecutive_same_class_entities(ner_tags, dataset_name)
        
        # Convert set to count for final stats
        stats['num_types'] = len(stats['entity_types'])
        
        return stats
    
    def calculate_advanced_stats(self, stats: Dict[str, Any]) -> Dict[str, float]:
        """Calculate additional meaningful statistics"""
        advanced = {}
        
        if stats['num_sentences'] > 0:
            advanced['avg_sentence_length'] = np.mean(stats['sentence_lengths'])
            advanced['std_sentence_length'] = np.std(stats['sentence_lengths'])
            advanced['avg_entities_per_sentence'] = np.mean(stats['entities_per_sentence'])
            advanced['entity_density'] = stats['num_entity_words'] / stats['num_words'] if stats['num_words'] > 0 else 0
        
        #if stats['entity_lengths']:
        #    advanced['avg_entity_length'] = np.mean(stats['entity_lengths'])
        #    advanced['std_entity_length'] = np.std(stats['entity_lengths'])
        
        if stats['num_entities'] > 0:
            advanced['compound_entity_ratio'] = stats['num_compound_entities'] / stats['num_entities']
            advanced['consecutive_same_class_ratio'] = stats['num_consecutive_same_class'] / stats['num_entities']
        
        return advanced
    
    def analyze_dataset(self, dataset_name: str, custom_tag_mapping: Dict[int, str] = None) -> Dict[str, Dict]:
        """Analyze complete dataset with all splits"""
        if custom_tag_mapping:
            self.dataset_tag_mappings[dataset_name] = custom_tag_mapping
            
        print(f"Analyzing dataset: {dataset_name}")
        
        # Load dataset
        dataset = self.load_hf_dataset(dataset_name)
        if not dataset:
            return {}
        
        results = {}
        
        # Analyze each split
        for split_name in dataset.keys():
            print(f"  Processing {split_name} split...")
            split_data = dataset[split_name]
            
            # Convert to list of dicts if needed
            if hasattr(split_data, 'to_list'):
                data_list = []
                for i in range(len(split_data)):
                    data_list.append(split_data[i])
            else:
                data_list = list(split_data)
            
            stats = self.analyze_split(data_list, dataset_name)
            advanced_stats = self.calculate_advanced_stats(stats)
            
            results[split_name] = {
                'basic_stats': stats,
                'advanced_stats': advanced_stats
            }
        
        return results
    
    def create_comparison_table(self, results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """Create a comparison table similar to the one in your image"""
        rows = []
        
        for dataset_name, dataset_results in results_dict.items():
            row = {'Dataset': dataset_name}
            
            for split_name in ['train', 'validation', 'test']:
                if split_name in dataset_results:
                    stats = dataset_results[split_name]['basic_stats']
                    
                    row[f'{split_name}_sentences'] = stats['num_sentences']
                    row[f'{split_name}_words'] = stats['num_words']
                    row[f'{split_name}_entities'] = stats['num_entities']
                    row[f'{split_name}_compound_entities'] = stats['num_compound_entities']
                    row[f'{split_name}_consecutive_same_class'] = stats['num_consecutive_same_class']
                    row[f'{split_name}_types'] = stats['num_types']
                else:
                    row[f'{split_name}_sentences'] = 0
                    row[f'{split_name}_words'] = 0
                    row[f'{split_name}_entities'] = 0
                    row[f'{split_name}_compound_entities'] = 0
                    row[f'{split_name}_consecutive_same_class'] = 0
                    row[f'{split_name}_types'] = 0
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def create_comparison_table_2(self, results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """Create a comparison table with transposed format matching the given image"""
        # Create column names for each dataset and split combination
        columns = []
        for dataset_name in results_dict.keys():
            for split_name in ['train', 'dev', 'test']:
                # Use 'dev' instead of 'validation' to match the image format
                actual_split = 'validation' if split_name == 'dev' else split_name
                columns.append(f"{dataset_name}_{split_name}")
        
        # Define the row metrics
        metrics = [
            ('sentences', 'num_sentences'),
            ('words', 'num_words'), 
            ('entities', 'num_entities'),
            ('compound_entities', 'num_compound_entities'),
            ('consecutive_same_class', 'num_consecutive_same_class'),
            ('types', 'num_types'),
            ('avg_sentence_length', 'avg_sentence_length'),
            ('avg_entities_per_sentence', 'avg_entities_per_sentence'),
            ('entity_density', 'entity_density')
        ]
        
        # Create the data dictionary
        data = {}
        
        for dataset_name, dataset_results in results_dict.items():
            for split_name in ['train', 'dev', 'test']:
                actual_split = 'validation' if split_name == 'dev' else split_name
                col_name = f"{dataset_name}_{split_name}"
                
                if actual_split in dataset_results:
                    stats = dataset_results[actual_split]['basic_stats']
                    stats_advanced = dataset_results[actual_split]['advanced_stats']
                    data[col_name] = [
                        stats['num_sentences'],
                        stats['num_words'], 
                        stats['num_entities'],
                        stats['num_compound_entities'],
                        stats['num_consecutive_same_class'],
                        stats['num_types'],
                        stats_advanced['avg_sentence_length'],
                        stats_advanced['avg_entities_per_sentence'],
                        stats_advanced['entity_density']
                    ]
                else:
                    data[col_name] = [0, 0, 0, 0, 0, 0]
        
        # Create DataFrame with metrics as index
        df = pd.DataFrame(data, index=[metric[0] for metric in metrics])
        
        return df
    
    def print_detailed_analysis(self, results: Dict[str, Dict], dataset_name: str):
        """Print detailed analysis for a single dataset"""
        print(f"\n{'='*60}")
        print(f"DETAILED ANALYSIS: {dataset_name}")
        print(f"{'='*60}")
        
        for split_name, split_results in results.items():
            basic = split_results['basic_stats']
            advanced = split_results['advanced_stats']
            
            print(f"\n{split_name.upper()} SPLIT:")
            print(f"  Sentences: {basic['num_sentences']:,}")
            print(f"  Words: {basic['num_words']:,}")
            print(f"  Entities: {basic['num_entities']:,}")
            print(f"  Compound entities: {basic['num_compound_entities']:,}")
            print(f"  Consecutive same-class entities: {basic['num_consecutive_same_class']:,}")
            print(f"  Entity types: {basic['num_types']}")
            
            if advanced:
                print(f"  Avg sentence length: {advanced.get('avg_sentence_length', 0):.1f}")
                print(f"  Avg entities per sentence: {advanced.get('avg_entities_per_sentence', 0):.2f}")
                print(f"  Entity density: {advanced.get('entity_density', 0):.4f}")
                print(f"  Compound entity ratio: {advanced.get('compound_entity_ratio', 0):.3f}")
            
            print(f"  Entity type distribution:")
            for entity_type, count in basic['entity_type_counts'].most_common():
                print(f"    {entity_type}: {count:,}")

# Example usage and main function
def main():
    analyzer = NERDatasetAnalyzer()
    
    # Example datasets - replace with your actual dataset names
    datasets = [
        #"conll2003",
        "ontonotes",
        "ibm-research/MedMentions-ZS",
        "alecocc/LegalNER-ZS",
        
        # Add your other datasets here
        # "your-legal-dataset",
    ]
    
    # Debug mode - set to True to see tag scheme information
    DEBUG_MODE = False
    
    if DEBUG_MODE:
        print("DEBUG MODE: Analyzing tag schemes...")
        for dataset_name in datasets:
            analyzer.debug_tag_scheme(dataset_name)
    
    
    tag_mappings = {}
    
    all_results = {}
    
    # Analyze each dataset
    for dataset_name in datasets:
        tag_mapping = tag_mappings.get(dataset_name, None)
        results = analyzer.analyze_dataset(dataset_name, tag_mapping)
        if results:
            all_results[dataset_name] = results
            analyzer.print_detailed_analysis(results, dataset_name)
    
    # Create comparison table
    if all_results:
        print(f"\n{'='*80}")
        print("COMPARISON TABLE")
        print(f"{'='*80}")
        
        comparison_df = analyzer.create_comparison_table_2(all_results)
        comparison_df = comparison_df.round(2)
        print(comparison_df.to_string(index=True))
        #print(comparison_df.T.to_string())
        
        # Save to CSV
        comparison_df.to_csv('ner_dataset_comparison.csv', index=True)
        print(f"\nComparison table saved to 'ner_dataset_comparison.csv'")

        comparison_df.to_latex("ner_dataset_comparison.tex", index=True, float_format="%.2f")
        print(f"\nComparison table LATEX saved to 'ner_dataset_comparison.text'")

        

if __name__ == "__main__":
    # Install required packages first:
    # pip install datasets pandas numpy
    
    main()


    

