import re
import pandas as pd
import nltk
import json
import os
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize
from datasets import Dataset
from tqdm import tqdm


def convert_to_bio_nltk(item):
    """
    Converts a given item with annotations into IOB2 format.

    Parameters:
    - item (dict): The dictionary containing the data and annotations.

    Returns:
    - list of tuples: Each tuple contains a token and its IOB2 tag.
    """
    text = item['text']
    annotations = item['annotations']

    # Initialize the tags for each character in the text as "O"
    tags = ['O'] * len(text)

    # Process each annotation
    for annotation in annotations:
        start = annotation['start']
        end = annotation['end']
        label = annotation['label']

        
        # Assign tags for the span
        tags[start] = f"B-{label}"  # Beginning tag
        for i in range(start + 1, end):
            tags[i] = f"I-{label}"  # Inside tag

    # Tokenize the text using NLTK's word_tokenize
    tokens = word_tokenize(text)

    # Map tokens to their start and end positions
    token_offsets = []
    offset = 0
    for token in tokens:
        start = text.find(token, offset)
        end = start + len(token)
        token_offsets.append((token, start, end))
        offset = end

    token_tags = []
    for token, token_start, token_end in token_offsets:
        # Determine the IOB tag for the token
        token_tag = 'O'
        for i in range(token_start, token_end):
            if tags[i] != 'O':
                token_tag = tags[i]
                break

        token_tags.append((token, token_tag))

    return token_tags


if __name__ == "__main__":

    input_path = 'output/data_creation/completions/Llama-3.1-8B-Instruct/completions.jsonl'
    output_path = 'output/data_creation/bio_converted'
    with open(input_path) as f:
        data = [json.loads(line) for line in f.readlines()]
    
    print("LEN DATA:", len(data))
    dataset = []
    os.makedirs(output_path, exist_ok=True)

    for item in tqdm(data):
        bio_annotations = convert_to_bio_nltk(item)

        tokens = [el[0] for el in bio_annotations]
        tags = [el[1] for el in bio_annotations]
            
        if sum([1 if tag != 'O' else 0 for tag in tags]) > 0:
            dataset.append({"tokens": tokens, "ner_tags": tags})

    ds = Dataset.from_pandas(pd.DataFrame(dataset))
    ds.to_json(output_path + "/pretrain_data.jsonl")