import json
import pandas as pd
import re
from datasets import Dataset, DatasetDict
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# Read JSON files
with open('NER_TRAIN_JUDGEMENT.json', 'r') as f:
    data_content = json.load(f)
with open('NER_TRAIN_PREAMBLE.json', 'r') as f2:
    data_content_2 = json.load(f2)
with open('NER_DEV_JUDGEMENT.json', 'r') as f:
    data_content_test = json.load(f)
with open('NER_DEV_PREAMBLE.json', 'r') as f2:
    data_content_test_2 = json.load(f2)

# Combine datasets
combined_data = data_content + data_content_2 + data_content_test + data_content_test_2

# Define COMPLETELY SEPARATE entity sets for zero-shot learning
train_entities = {'COURT', 'PETITIONER', 'RESPONDENT', 'LAWYER', 'PROVISION', 'OTHER_PERSON'}
validation_entities = {'JUDGE', 'ORG', 'STATUTE'}
test_entities = {'GPE', 'PRECEDENT', 'CASE_NUMBER', 'WITNESS'}

# Verify no overlap between entity sets
assert len(train_entities & validation_entities) == 0, "Train and validation entities overlap!"
assert len(train_entities & test_entities) == 0, "Train and test entities overlap!"
assert len(validation_entities & test_entities) == 0, "Validation and test entities overlap!"

print(f"Train entities: {train_entities}")
print(f"Validation entities: {validation_entities}")
print(f"Test entities: {test_entities}")

def get_entities_in_item(item):
    """Extract all unique entity types from an item's annotations."""
    annotations = item['annotations'][0]['result']
    return {entity['value']['labels'][0] for entity in annotations}

def categorize_data_strict(combined_data, train_entities, validation_entities, test_entities):
    """
    Strictly categorize data ensuring NO entity overlap between splits.
    An item goes to a split only if ALL its entities belong to that split's entity set.
    """
    train_data = []
    validation_data = []
    test_data = []
    mixed_data = []  # Items with entities from multiple splits
    
    for item in tqdm(combined_data, desc="Categorizing data"):
        item_entities = get_entities_in_item(item)
        
        # Check if all entities belong to one specific split
        if item_entities.issubset(train_entities):
            train_data.append(item)
        elif item_entities.issubset(validation_entities):
            validation_data.append(item)
        elif item_entities.issubset(test_entities):
            test_data.append(item)
        else:
            # Item has entities from multiple splits or unknown entities
            mixed_data.append({
                'item': item,
                'entities': item_entities,
                'train_overlap': item_entities & train_entities,
                'val_overlap': item_entities & validation_entities,
                'test_overlap': item_entities & test_entities
            })
    
    print(f"\nData distribution:")
    print(f"Train: {len(train_data)} items")
    print(f"Validation: {len(validation_data)} items")
    print(f"Test: {len(test_data)} items")
    print(f"Mixed/Unknown: {len(mixed_data)} items")
    
    if mixed_data:
        print(f"\nWarning: {len(mixed_data)} items have mixed entity types and will be excluded")
        # Optionally print first few mixed items for debugging
        for i, mixed_item in enumerate(mixed_data[:3]):
            print(f"Mixed item {i+1}: entities={mixed_item['entities']}")
    
    return train_data, validation_data, test_data

def convert_to_iob2_fixed(item, target_entities):
    """
    Converts annotations to IOB2 format with proper token alignment.
    Only processes entities that are in the target_entities set.
    """
    text = item['data']['text']
    annotations = item['annotations'][0]['result']

    # Tokenize first to get proper token boundaries
    tokens = word_tokenize(text)
    
    # Find token positions in original text
    token_positions = []
    current_pos = 0
    
    for token in tokens:
        # Find token in text starting from current position
        token_start = text.find(token, current_pos)
        if token_start == -1:
            # Handle special cases where tokenizer might split differently
            # For now, use approximate position
            token_start = current_pos
        
        token_end = token_start + len(token)
        token_positions.append((token, token_start, token_end))
        current_pos = token_end
    
    # Initialize all tokens as 'O'
    token_tags = ['O'] * len(tokens)
    
    # Process annotations
    for annotation in annotations:
        start = annotation['value']['start']
        end = annotation['value']['end']
        label = annotation['value']['labels'][0]
        
        # Only process entities that are in our target set
        if label not in target_entities:
            continue
        
        # Find which tokens overlap with this annotation
        first_token_idx = None
        for i, (token, token_start, token_end) in enumerate(token_positions):
            # Check if token overlaps with annotation
            if token_start < end and token_end > start:
                if first_token_idx is None:
                    first_token_idx = i
                    token_tags[i] = f"B-{label}"
                else:
                    token_tags[i] = f"I-{label}"
    
    return [(token, tag) for token, tag in zip(tokens, token_tags)]

def create_dataset_split(data_items, target_entities, output_file=None):
    """Create dataset split with proper IOB2 conversion."""
    hf_dataset = []
    filtered_count = 0
    filtered_html = 0
    
    if output_file:
        # Clear the file first
        with open(output_file, 'w') as f:
            pass
    
    for item in tqdm(data_items, desc=f"Processing {len(data_items)} items"):

        # Find all HTML tags with their positions
        html_pattern = r'<[^>]+>'
        html_matches = list(re.finditer(html_pattern, item['data']['text']))
        if html_matches:
            #print(html_matches)
            filtered_html += 1
            continue
    
        iob_annotation = convert_to_iob2_fixed(item, target_entities)
        
        # Check if all tags are "O" (no entities)
        tags_only = [tag for token, tag in iob_annotation]
        if all(tag == 'O' for tag in tags_only):
            filtered_count += 1
            continue  # Skip samples with no target entities
        
        hf_item = {"tokens": [], "ner_tags": []}
        
        if output_file:
            with open(output_file, 'a', encoding='utf-8') as f:
                for token, tag in iob_annotation:
                    f.write(f"{token} {tag}\n")
                    hf_item['tokens'].append(token)
                    hf_item['ner_tags'].append(tag)
                f.write("\n")  # Separate sentences
        else:
            for token, tag in iob_annotation:
                hf_item['tokens'].append(token)
                hf_item['ner_tags'].append(tag)
        
        hf_dataset.append(hf_item)
    
    if filtered_count > 0:
        print(f"  Filtered out {filtered_count} samples with no target entities")
    
    if filtered_html > 0:
        print(f"  Filtered out {filtered_html} samples with HTML tags")
    
    return hf_dataset

# Main execution
if __name__ == "__main__":
    # Categorize data with strict separation
    train_data, validation_data, test_data = categorize_data_strict(
        combined_data, train_entities, validation_entities, test_entities
    )
    
    # Verify no entity leakage
    def verify_no_leakage(data, expected_entities, split_name):
        all_entities = set()
        for item in data:
            item_entities = get_entities_in_item(item)
            all_entities.update(item_entities)
        
        unexpected = all_entities - expected_entities
        if unexpected:
            print(f"ERROR: {split_name} split has unexpected entities: {unexpected}")
        else:
            print(f"✓ {split_name} split is clean with entities: {all_entities}")
        
        return len(unexpected) == 0
    
    # Verify each split
    train_clean = verify_no_leakage(train_data, train_entities, "Train")
    val_clean = verify_no_leakage(validation_data, validation_entities, "Validation")
    test_clean = verify_no_leakage(test_data, test_entities, "Test")
    
    if not all([train_clean, val_clean, test_clean]):
        print("ERROR: Entity leakage detected! Please check your data.")
        exit(1)
    
    # Create datasets
    print("\nCreating datasets...")
    print("Note: Samples with no target entities (all 'O' tags) will be filtered out")
    hf_dataset_train = create_dataset_split(train_data, train_entities, 'gliner_train_iob.txt')
    hf_dataset_validation = create_dataset_split(validation_data, validation_entities, 'gliner_validation_iob.txt')
    hf_dataset_test = create_dataset_split(test_data, test_entities, 'test_iob.txt')
    
    # Print final counts
    print(f"\nFinal dataset sizes after filtering:")
    print(f"Train: {len(hf_dataset_train)} samples")
    print(f"Validation: {len(hf_dataset_validation)} samples") 
    print(f"Test: {len(hf_dataset_test)} samples")
    
    # Convert to HuggingFace datasets
    hf_legalner_train = Dataset.from_pandas(pd.DataFrame(hf_dataset_train))
    hf_legalner_validation = Dataset.from_pandas(pd.DataFrame(hf_dataset_validation))
    hf_legalner_test = Dataset.from_pandas(pd.DataFrame(hf_dataset_test))
    
    # Create dataset dictionary
    hf_legalner = DatasetDict({
        'train': hf_legalner_train,
        'validation': hf_legalner_validation,
        'test': hf_legalner_test
    })
    
    print(f"\nFinal dataset:")
    print(hf_legalner)
    
    # Push to hub
    hf_legalner.push_to_hub("alecocc/LegalNER-ZS", private=True)
    
    print("\n✓ Zero-shot NER dataset created successfully!")
    print("✓ No entity overlap between splits")
    print("✓ IOB2 format conversion completed")
    print("✓ Dataset pushed to HuggingFace Hub")


"""
We process the zero-shot NER dataset by partitioning entity types into three disjoint sets: training entities (\texttt{COURT}, \texttt{PETITIONER}, \texttt{RESPONDENT}, \texttt{LAWYER}, \texttt{PROVISION}, \texttt{OTHER\_PERSON}), validation entities (\texttt{JUDGE}, \texttt{ORG}, \texttt{STATUTE}), and test entities (\texttt{GPE}, \texttt{PRECEDENT}, \texttt{CASE\_NUMBER}, \texttt{WITNESS}). 
Documents are strictly categorized such that each split contains only samples with entities from its designated set, ensuring zero entity overlap between splits.
We convert annotations to IOB2 format using NLTK tokenization with proper token-span alignment, filtering out samples containing HTML tags or lacking target entities. 
This approach creates a true zero-shot evaluation scenario where the model encounters completely unseen entity types during validation and testing phases.
"""