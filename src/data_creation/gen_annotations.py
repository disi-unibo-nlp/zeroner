import torch
import json
import os
import logging
import pandas as pd
import numpy as np 
import json
import yaml
import ast 

from dotenv import load_dotenv
from tqdm import tqdm
from datasets import load_dataset, Dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, HfArgumentParser
from huggingface_hub import login
from typing import Optional
from dataclasses import dataclass, field
from collections import Counter, defaultdict 
from huggingface_hub import hf_hub_download
from hashlib import sha512

import re
import sys
import io
import traceback
import multiprocessing
import signal
import warnings

# Load variables from the .env file
load_dotenv()

 # 11:03:46 llm_engine.py:161] Initializing an LLM engine (v0.5.0) with config: model='Qwen/Qwen2.5-Math-7B-Instruct', speculative_config=None, tokenizer='Qwen/Qwen2.5-Math-7B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, rope_scaling=None, rope_theta=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=4096, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=True, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), seed=0, served_model_name=Qwen/Qwen2.5-Math-7B-Instruct)

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="meta-llama/Llama-3.1-8B-Instruct", metadata={"help": "model's HF directory or local path"})
    input_path: Optional[str] = field(default="data/pretrain/processed/data_processed_3.jsonl")
    out_dir: Optional[str] =  field(default="./output/data_creation", metadata={"help": "outputs directory"})
    max_samples: Optional[int] = field(default=-1, metadata={"help": "Maximum number of data to process in train set. Default is -1 to process all data."})
    start_idx: Optional[int] = field(default=0, metadata={"help": "Index of first prompt to process."})
    batch_size: Optional[int] = field(default=128, metadata={"help": "Maximum number of data to process per batch."})
    cache_dir: Optional[str] =  field(default=None, metadata={"help": "cache dir to store model weights"})
    max_model_len: Optional[int] = field(default=1024, metadata={"help": "Maximum input sequence length"})
    top_p: Optional[float] = field(default=1.0, metadata={"help": "Top p sampling."})
    n_out_sequences: Optional[int] = field(default=1, metadata={"help": "Number of generated sequences per instance"})
    temperature: Optional[float] = field(default=0.0, metadata={"help": "Sampling temperature parameter"})
    n_gpus: Optional[int] = field(default=1, metadata={"help": "Number of gpus to use for inference."})
    

if __name__ == "__main__":

    HF_TOKEN = os.getenv("HF_TOKEN")
    login(token=HF_TOKEN)
    
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename="output/logs/annotations.log",
                        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    # parse input args
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    sampling_params = SamplingParams(
        n=args.n_out_sequences, 
        temperature=args.temperature, 
        top_p=args.top_p, 
        max_tokens=1024, 
        seed=0
    )

    
    llm = LLM(
        model=args.model_name,
        gpu_memory_utilization=.95,
        dtype="half" if "awq" in args.model_name.lower() else "auto",
        quantization="awq" if "awq" in args.model_name.lower() else None,
        #download_dir=args.cache_dir,
        enforce_eager=True,
        max_model_len=args.max_model_len if args.max_model_len > 0 else None,
        trust_remote_code=True,
        tensor_parallel_size=args.n_gpus,
    )

    with open(args.input_path) as f:
        dataset = [json.loads(line) for line in f.readlines()]
    dataset = Dataset.from_pandas(pd.DataFrame(dataset))
    
    if args.max_samples > 0: # to use for debug
        dataset = dataset.select(range(args.start_idx, args.max_samples))
    
    if args.start_idx > 0 and args.max_samples < 0: # to use for debug
        dataset = dataset.select(range(args.start_idx, len(dataset)))

    SYSTEM_PROMPT = """You are an expert named entity annotator.
Annotate the given text with entity mentions and their corresponding labels. Then, define the most appropriate domain for the text.

Requirements:
Report annotations as a list of tuples in the following format: [('entity', 'LABEL'), ...]

Each entity must include:
- The exact string as it appears in the text.
- The corresponding entity label.

Ensure that:
- Labels do not overlap.
- Each entity is assigned to only one label.
- Labels are uppercased.

Output following this schema:
Annotations: [list of tuples]
Domain: {domain of the text}
"""

    prompts = []
    for i, item in enumerate(tqdm(dataset, total=len(dataset))):
        
        if "llama-3.1" in args.model_name.lower():
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Text: {item['text']}"}
            ]
        
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        input_prompt = {}
        for key in item.keys():
            input_prompt[key] = item[key]

        prompt_footprint_yaml = yaml.dump(item, sort_keys=True)
        prompt_id = sha512(prompt_footprint_yaml.encode()).hexdigest()
   
        input_prompt['prompt'] = prompt
        input_prompt['id'] = prompt_id

        prompts.append(input_prompt)
        #prompts.append((item['id'], text, messages))

    new_prompts = []
    for prompt in prompts:
        if prompt['actual_tokens'] <= 300:
            new_prompts.append(prompt)

    # save first 5 prompts to txt file
    os.makedirs(args.out_dir + "/prompts", exist_ok=True)
    n_prompts_to_stamp = 100 if args.max_samples > 5 else args.max_samples
   
    with open(args.out_dir + '/prompts/example_prompts.txt', 'w') as f:
        for i in range(n_prompts_to_stamp):
            f.write(f"ID: {new_prompts[i]['id']}\n")
            f.write(new_prompts[i]['prompt'])
            f.write(f"LEN TEXT: {len(tokenizer.encode(new_prompts[i]['text']))}\n")
            f.write(f"LEN PROMPT: {len(tokenizer.encode(new_prompts[i]['prompt']))}\n")
            f.write("*"*100+'\n')
    
    batches = [new_prompts[i:i+args.batch_size] for i in range(0, len(new_prompts), args.batch_size)]
    batches = batches[153:]
    logger.info(f"Number of START prompts: {len(prompts)}")
    logger.info(f"Number of prompts: {len(new_prompts)}")
    logger.info(f"Number of batches: {len(batches)}")
    logger.info(f"Number of prompts in each batch: {len(batches[0])}")

    

    model_name = args.model_name.split("/")[-1]
    os.makedirs(args.out_dir + f"/completions/{model_name}", exist_ok=True)
    for id_batch, batch in enumerate(tqdm(batches)):

        ids = [el['id'] for el in batch]
        input_prompts = [el['prompt'] for el in batch]
        texts = [el['text'] for el in batch]

        outputs = llm.generate(input_prompts, sampling_params, use_tqdm=False)

        for id_out, out in enumerate(outputs):
            completions = [o.text.strip() for o in out.outputs]
            final_annotations = []
            for completion in completions:
                start_annotation = completion.find("Annotations:") + len("Annotations:")
                end_annotation = completion.rfind("Domain:")
                annotation = completion[start_annotation:end_annotation].strip()
                domain = completion[completion.rfind("Domain:"):].replace("Domain:", "").strip()
                
                # try:
                #     annotation = ast.literal_eval(annotation)
                #     annotation = [(str(item[0]), str(item[1])) for item in annotation if isinstance(item, tuple) and len(item) == 2]
                # except (ValueError, SyntaxError):
                #     annotation = []
                
                try:
                    annotation = ast.literal_eval(annotation)
                    if isinstance(annotation, (list, tuple)):  # Ensure annotation is iterable
                        annotation = [
                            (str(item[0]), str(item[1])) 
                            for item in annotation 
                            if isinstance(item, tuple) and len(item) == 2 and all(isinstance(subitem, (str, int, float)) for subitem in item)
                        ]
                    else:
                        annotation = []  # Fallback to an empty list if not iterable
                except (ValueError, SyntaxError, TypeError):
                    annotation = [] 
                                    
                input_text = texts[id_out]
                for mention, label in annotation:
                    if mention in input_text:
                        start_mention = input_text.find(mention)
                        end_mention = start_mention + len(mention)
                        final_annotations.append({"start": start_mention, "end": end_mention, "mention": mention, "label": label.upper()})
            
                with open(args.out_dir + f"/completions/{model_name}/completions.jsonl", 'a') as f:
                    json.dump({"annotations": final_annotations, "domain": domain, "text": texts[id_out], "completion": completion, "id": ids[id_out] }, f, ensure_ascii=False)
                    f.write('\n')
    