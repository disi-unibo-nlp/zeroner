import os
import yaml
import json
import yaml
import random
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

from datasets import load_dataset
from transformers import BertTokenizer
from nltk.tokenize import sent_tokenize
from hashlib import sha512
from tqdm import tqdm 
from multiprocessing import Pool, cpu_count
from filelock import FileLock


def generate_max_tokens(min_tokens=30, max_tokens=300, skew_threshold=100):
    """
    Generate a max_tokens value with a higher probability for values above the skew_threshold.

    Args:
        min_tokens (int): Minimum token limit.
        max_tokens (int): Maximum token limit.
        skew_threshold (int): Value above which probabilities are higher.

    Returns:
        int: A randomly generated max_tokens value.
    """
    if random.random() < 0.7:  # 70% chance to favor values above skew_threshold
        return random.randint(skew_threshold, max_tokens)
    return random.randint(min_tokens, max_tokens)


def chunk_article(article, max_tokens):
    """
    Splits an article into chunks using BERT tokenizer.
    Ensures no chunk exceeds the dynamically chosen max_tokens allowed.

    Args:
        article (dict): The text and metadata of the article to be chunked.

    Returns:
        List[dict]: List of text chunks with metadata.
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    # Split the article into sentences
    article_text = article['text']
    subset = article['meta']['pile_set_name']
    sentences = sent_tokenize(article_text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        # Tokenize the sentence and calculate its length
        sentence_tokens = tokenizer.tokenize(sentence)
        sentence_length = len(sentence_tokens)

        # If a single sentence is too long, truncate it
        if sentence_length > max_tokens:
            # Split the long sentence into smaller chunks
            for i in range(0, sentence_length, max_tokens):
                chunk_text = tokenizer.convert_tokens_to_string(sentence_tokens[i:i + max_tokens])
                if chunk_text.strip():
                    chunks.append({"text": chunk_text, "subset": subset, "actual_tokens": len(sentence_tokens[i:i + max_tokens]), "max_tokens": max_tokens})
            continue

        # Check if adding the sentence exceeds the current max_tokens limit
        if current_length + sentence_length > max_tokens:
            # Finalize the current chunk
            chunk_text = tokenizer.convert_tokens_to_string(current_chunk)
            if chunk_text.strip():
                chunks.append({"text": chunk_text, "subset": subset, "actual_tokens": current_length, "max_tokens": max_tokens})

            # Start a new chunk
            current_chunk = sentence_tokens
            current_length = sentence_length
        else:
            # Add the sentence to the current chunk
            current_chunk.extend(sentence_tokens)
            current_length += sentence_length

    # Add the last chunk if it has content
    if current_chunk:
        chunk_text = tokenizer.convert_tokens_to_string(current_chunk)
        if chunk_text.strip():
            chunks.append({"text": chunk_text, "subset": subset, "actual_tokens": len(current_chunk), "max_tokens": max_tokens})

    return chunks

# def chunk_article(article, max_tokens):
#     """
#     Splits an article into chunks using BERT tokenizer.
#     First splits the article into sentences and then combines sentences to fit the dynamically chosen max tokens allowed.

#     Args:
#         article (str): The text to be chunked.
#         min_tokens (int): Minimum token limit.
#         max_tokens (int): Maximum token limit.
#         skew_threshold (int): Value above which probabilities are higher for max_tokens.

#     Returns:
#         List[str]: List of text chunks.
#     """
#     tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

#     # Split the article into sentences
#     article_text = article['text']
#     subset = article['meta']['pile_set_name']
#     sentences = sent_tokenize(article_text)

#     chunks = []
#     current_chunk = []
#     current_length = 0

#     for sentence in sentences:
#         # Tokenize the sentence and calculate its length
#         sentence_tokens = tokenizer.tokenize(sentence)
#         sentence_length = len(sentence_tokens)

#         # Check if adding the sentence exceeds the current max_tokens limit
#         if current_length + sentence_length > max_tokens:
#             # Finalize the current chunk
#             chunk_text = tokenizer.convert_tokens_to_string(current_chunk)
#             if chunk_text.strip():
#                 chunks.append({"text": chunk_text, "subset": subset, "actual_tokens": current_length, "max_tokens": max_tokens})

#             # Start a new chunk and regenerate max_tokens for the next chunk
#             current_chunk = sentence_tokens
#             current_length = sentence_length
#         else:
#             # Add the sentence to the current chunk
#             current_chunk.extend(sentence_tokens)
#             current_length += sentence_length

#     # Add the last chunk if it has content
#     if current_chunk:
#         chunk_text = tokenizer.convert_tokens_to_string(current_chunk)
#         if chunk_text.strip():
#             chunks.append({"text": chunk_text, "subset": subset, "actual_tokens": len(current_chunk), "max_tokens": max_tokens})

#     return chunks

def process_article(article):
    """
    Processes a single article, chunks it, and computes unique IDs for each chunk.

    Args:
        article (str): The text of the article to process.

    Returns:
        List[dict]: List of chunk dictionaries with unique IDs.
    """
    max_tokens = generate_max_tokens(min_tokens=30, max_tokens=300, skew_threshold=100)
    chunks = chunk_article(article, max_tokens=max_tokens)

    return chunks

def process_and_write(article_data):
    """
    Processes an article and writes the results to the output file.

    Args:
        article_data (tuple): A tuple containing the article text and output file path.
    """
    article, output_file = article_data
    processed_chunks = process_article(article)
    write_chunks_to_file(processed_chunks, file_path=output_file)

def write_chunks_to_file(chunk_dicts, file_path="data.jsonl"):
    """
    Writes the chunk dictionaries to a JSONL file with file-locking to prevent race conditions.

    Args:
        chunk_dicts (List[dict]): List of chunk dictionaries to write.
        file_path (str): Path to the output JSONL file.
    """
    lock_path = f"{file_path}.lock"  # Lock file path
    with FileLock(lock_path, timeout=10):  # Lock timeout set to 10 seconds
        with open(file_path, 'a', encoding='utf-8') as f:
            for chunk in chunk_dicts:
                json.dump(chunk, f, ensure_ascii=False)
                f.write("\n")

if __name__ == "__main__":
    output_dir = "data/pretrain/tokenized"
    output_file = output_dir + "/data_5.jsonl"
    os.makedirs(output_dir, exist_ok=True)
    # Prepare data for multiprocessing

    
    dataset = load_dataset('monology/pile-uncopyrighted', split="train", streaming=True)

    data_selected = dataset.take(50000)

    new_data = []

    for item in data_selected:
        new_data.append(item)

    articles_with_output = [(article, output_file) for article in new_data]

    # Define the number of processes to use
    num_processes = min(cpu_count(), len(new_data))
    print("Number of processes:", num_processes)

    with Pool(processes=num_processes) as pool:
        # Process articles and write incrementally
        list(tqdm(pool.imap_unordered(process_and_write, articles_with_output), total=len(new_data)))
