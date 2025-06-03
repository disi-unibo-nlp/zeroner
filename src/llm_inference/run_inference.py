import torch
import json
import os
import json
import logging
import pandas as pd
import numpy as np 
import json
from dotenv import load_dotenv
from tqdm import tqdm
from datasets import load_dataset, Dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, HfArgumentParser
from huggingface_hub import login
from typing import Optional
from dataclasses import dataclass, field
from collections import Counter, defaultdict 


with open('data/entity_type_mapping/zeroner_descr_map.json', 'r') as f:
    DESCRIPTIONS = json.load(f)

name2id = {
    "MedMentions-ZS": "medmentions",
    "OntoNotes-ZS": "ontonotes",
    "LegalNER-ZS": "legalner"
}


# for medmentions label mapping
id2tag =  {'T058': "Health_Care_Activity", "T062": "Research_Activity", "T037": "Injury_or_Poisoning",
            "T038": "Biologic_Function", "T005": "Virus", "T007": "Bacterium", "T204": "Eukaryote",
            "T017": "Anotomical_Structure", "T074": "Medical_Device", "T031": "Body_Substance", "T103": "Chemical",
            "T168": "Food", "T201": "Clinical_Attribute", "T033": "Finding", "T082": "Spatial_Concept",
            "T022": "Body_System", "T091": "Biomedical_Occupation_or_Discipline", "T092": "Organization",
            "T097": "Professional_or_Occupational_Group", "T098": "Population_Group", "T170": "Intellectual_Product",
            "NEG": "NEG"}

def medmentions_type_dict_inv():
    """
    Inverse mapping of MedMentions entity types to their corresponding IDs.
    
    Returns:
        dict: A dictionary where keys are entity type IDs and values are the corresponding entity type names.
    """
    return {v: k for k, v in id2tag.items()}

def convert_medmentions(example):
        tokens = example['tokens']
        ner_tags = example['ner_tags']
        example = [ (x[:2] + id2tag[x[2:]]).upper() if x != 'O' else 'O' for x in ner_tags]
        return {'tokens': tokens, 'ner_tags': example}
    

def read_dataset(file_path, dataset):
    with open(file_path, "r") as f:
        lines = f.readlines()
    sentences = []
    labels = []
    sentence = []
    label = []

    for line in lines:
        if line.strip() == "":
            if sentence:
                sentences.append(sentence)
                labels.append(label)
                sentence = []
                label = []
        else:
            
            word, tag = line.strip().split()
            sentence.append(word)
            label.append(tag)
    if sentence:
        sentences.append(sentence)
        labels.append(label)

    return sentences, labels

def list_all_labels(file_path):
    tags = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.strip():
                words = line.split()
            if len(words) > 1:
                if words[1] != 'O' and words[1][2:] not in tags:
                    if words[1] not in tags:
                        tags.append(words[1][2:].upper())

    tags = list(set(tags))
    tags = sorted(tags, key=lambda label: (tuple() if label == "O" else tuple(label.split('-', 1)[::-1])))

    return tags

#microsoft/Phi-3.5-mini-instruct
#meta-llama/Llama-3.1-8B-Instruct
#Qwen/Qwen2.5-7B-Instruct
@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="Qwen/Qwen2.5-7B-Instruct", metadata={"help": "model's HF directory or local path"})
    out_preds_dir: Optional[str] =  field(default="out/llm/predictions", metadata={"help": "outputs directory"})
    max_samples: Optional[int] = field(default=-1, metadata={"help": "Maximum number of data to process in train set. Default is -1 to process all data."})
    batch_size: Optional[int] = field(default=100, metadata={"help": "Maximum number of data to process per batch."})
    cache_dir: Optional[str] =  field(default="./models", metadata={"help": "cache dir to store model weights"})
    datasets: Optional[str] =  field(default="OntoNotes-ZS,MedMentions-ZS,LegalNER-ZS", metadata={"help": "cache dir to store model weights"})
    w_descriptions: Optional[bool] =  field(default=False, metadata={"help": "cache dir to store model weights"})
    hf_username: Optional[str] =  field(default="", metadata={"help": "username for downloading dataset from HuggingFace repo."})

if __name__ == "__main__":

    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    login(token=HF_TOKEN)
    
    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
                        filename="out_fail.log",
                        filemode='w')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    # parse input args
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if "llama" in args.model_name.lower():
        terminators = [
            tokenizer.eos_token,
            "<|eot_id|>"
        ]
    else:
        terminators = None

    sampling_params = SamplingParams(
        n=1, 
        temperature=0.0, 
        top_p=1.0, 
        max_tokens=1048, 
        #use_beam_search=False,
        stop=terminators,
        seed=0
    )
    
    llm = LLM(
        model=args.model_name,
        gpu_memory_utilization=.9,    
        max_model_len=2048,
        dtype="half" if "awq" in args.model_name.lower() else "auto",
        quantization="awq_merlin" if "awq" in args.model_name.lower() else None,
        #download_dir= "../medgnp/hf_cache",#"../../../preprocess/models",#"../chatbot/models",#"../preprocess/models",# #args.cache_dir,
        enforce_eager=True,
        trust_remote_code=True
    )
    
    MODEL_NAME = args.model_name.split("/")[-1]
    #SYS_PROMPT= "You are an expert entity classifier."
    instruction = """You are an expert entity classifier. You have to identify the entities within the given sentence that belong to one of the specified labels. 
There are no overlapped entities, so each word can belong to only one label. 
You can use only the possible labels provided. No other label is allowed.

Provide output in the following format: 
Return a list, marked with square brackets '[' and ']', containing string tuples. 
Each tuple should follow the pattern: ("entity", "label"). 
Prefix the entire list with '### entities:'. 
For example: 
### entities: [("entity 1", "label of entity 1"), ("entity 2", "label of entity 2"), ...]

If no entities are found, return an empty list like this: 
### entities: []

Don't add further information.
"""

    PROMPT_TEMPLATE = """Possible labels: {labels}

Sentence: {sentence}
"""
    dataset_list = args.datasets.split(",")#"MedMentions-ZS"]#"LegalNER-ZS", "OntoNotes-ZS",]#, #"CrossNER_literature", "CrossNER_music", "CrossNER_politics", "CrossNER_science","mit-movie", "mit-restaurant"]

    for dataset in dataset_list:
        #sentences,_ = read_dataset(f"./Dataset/{dataset}/test.bio.txt")
        if args.hf_username.strip():
            username = args.hf_username
            dataset_test = load_dataset(username+dataset, split="test")
        else:
            sentences, labels = read_dataset(f"data/dataset/{dataset}/test.bio.txt", dataset=dataset)
            dataset_test = Dataset.from_pandas(pd.DataFrame([{"tokens": s, "ner_tags": l} for s, l in zip(sentences, labels)]))
        
        if "medmentions" in dataset.lower():
            dataset_test = dataset_test.map(convert_medmentions)

        w_descriptions = args.w_descriptions
        sentences = dataset_test['tokens']
        sentences = [" ".join(s) for s in sentences]

        #list_of_labels = list_all_labels(f"./Dataset/{dataset}/test.bio.txt")
        # if dataset == "MedMentions-ZS":
        #     list_of_labels = list(set([medmentions_type_dict_inv()[l[2:]] for labels in dataset_test['ner_tags'] for l in labels if l != 'O']))
        #     list_of_labels = [lb.upper() for lb in list_of_labels]
        # else:
        list_of_labels = list(set([l[2:].upper() for labels in dataset_test['ner_tags'] for l in labels if l != 'O']))
        
        print("ALL LABELS:", list_of_labels)
        prompts = []
        for sentence in sentences:
            
            hint = ""
            if not w_descriptions:
                
                if dataset == "OntoNotes-ZS":
                    hint = "Consider that FAC corresponds to Facility and LOC corresponds to Location." 
                if dataset == "LegalNER-ZS":
                    hint = "Consider that GPE corresponds to Geopolitical Entity."
                prompt = PROMPT_TEMPLATE.format(sentence=sentence, labels=list_of_labels)
            else:
                list_of_labels_names = [el for el in list_of_labels]
                print("label name:", list_of_labels_names)
                list_of_labels_string = ""
                for name in list_of_labels_names:
                    list_of_labels_string += (name + ": " + DESCRIPTIONS[name2id[dataset]]['test'][name] + "\n")
                list_of_labels_string = "\n" + list_of_labels_string.strip()
                prompt = PROMPT_TEMPLATE.format(sentence=sentence, labels=list_of_labels_string)

            

            if "llama" in MODEL_NAME.lower() or "qwen2.5" in MODEL_NAME.lower() or "mistral" in MODEL_NAME.lower():
                prompt = [
                    {"role": "system", "content": f"{instruction}"},
                    {"role": "user", "content": f"{prompt}\n{hint}"},
                ]
            else:
                prompt = [
                    {"role": "user", "content": instruction + "\n" + prompt + "\n" +hint},
            ] 
            prompts.append(
                (tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True), 
                {"dataset": dataset, "sentence": sentence})
            )
    
        batches = [prompts[i:i+args.batch_size] for i in range(0, len(prompts), args.batch_size)]

        os.makedirs(f"out/llm/prompts/{MODEL_NAME}", exist_ok=True)
        os.makedirs("out/llm/out", exist_ok=True)
        # save first prompt to txt file
        with open(f'./out/llm/prompts/{MODEL_NAME}/{dataset}_desc_{w_descriptions}.txt', 'w') as f:
            for i in range(5):
                f.write(prompts[i][0])
                f.write("*"*100+'\n')

        for id_batch, batch in enumerate(tqdm(batches)):
            
            batch_prompts = [el[0] for el in batch]
            batch_items = [el[1] for el in batch]

            outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)

            for id_out, out in enumerate(outputs):

                prompt = out.prompt
                for k, o in enumerate(out.outputs):
                    if '###entities:' in o.text:
                        split = o.text.split('###entities:')
                    if '### entities:' in o.text :
                        split = o.text.split('### entities:')
                    prediction = split[-1].strip()
                    try:
                        completion_filename = f"out/llm/predictions/{MODEL_NAME}/{dataset}/completion.txt"
                        if w_descriptions:
                            completion_filename = completion_filename.replace(".txt","")
                            completion_filename += "_w_descriptions.txt"
                        with open(completion_filename, 'a') as f:
                            f.write(f"{prediction}\n")
                            f.write(f"********************************************\n")
                        prediction = eval(prediction.split("\n")[0])
                        prediction = [(pred[0], pred[1].upper()) for pred in prediction if pred[1].upper() in list_of_labels]
                       
                    except Exception as e: 
                        print(e)
                        prediction = []

                    dataset = batch_items[id_out]['dataset']
                    sentence = batch_items[id_out]['sentence']

                    os.makedirs(f"out/llm/predictions/{MODEL_NAME}/{dataset}/", exist_ok=True)
                    out_filename = f'out/llm/predictions/{MODEL_NAME}/{dataset}/predictions.txt'
                    if w_descriptions:
                        out_filename = out_filename.replace('.txt','')
                        out_filename += '_w_descriptions.txt'
                    
                    with open(out_filename, 'a') as f:
                        f.write(f"{prediction}\n")
                        #json.dump(prediction, f, ensure_ascii=False)
                        #f.write("\n")