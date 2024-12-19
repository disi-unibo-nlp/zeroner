import ast 
import os
import json
from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from seqeval.metrics import f1_score, recall_score, precision_score, classification_report
#from zshot.evaluation.dataset import load_ontonotes_zs, load_medmentions_zs

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="Qwen/Qwen2.5-7B-Instruct", metadata={"help": "model's HF directory or local path"})
    datasets: Optional[str] =  field(default="OntoNotes-ZS,MedMentions-ZS,LegalNER-ZS", metadata={"help": "outputs directory"})
    w_descriptions: Optional[bool] =  field(default=False, metadata={"help": "cache dir to store model weights"})

# for medmentions label mapping
def medmentions_type_dict_inv():
    return {'T058': "Health_Care_Activity", "T062": "Research_Activity", "T037": "Injury_or_Poisoning",
            "T038": "Biologic_Function", "T005": "Virus", "T007": "Bacterium", "T204": "Eukaryote",
            "T017": "Anotomical_Structure", "T074": "Medical_Device", "T031": "Body_Substance", "T103": "Chemical",
            "T168": "Food", "T201": "Clinical_Attribute", "T033": "Finding", "T082": "Spatial_Concept",
            "T022": "Body_System", "T091": "Biomedical_Occupation_or_Discipline", "T092": "Organization",
            "T097": "Professional_or_Occupational_Group", "T098": "Population_Group", "T170": "Intellectual_Product",
            "NEG": "NEG"}

def label_tokens(sentence, entities):
    sentence = [s.lower() for s in sentence]
    # Initialize all labels as "O"
    labels = ['O'] * len(sentence)

    # Convert sentence to a single string for easier substring matching
    sentence_str = ' '.join(sentence)
    # Track which tokens have been labeled
    labeled_positions = set()

    for entity, label in entities:
        # Split the entity into tokens
        if not isinstance(entity, str):
            entity = str(entity)

        entity_tokens = entity.split()
        entity_tokens = [token.lower() for token in entity_tokens]

        entity_length = len(entity_tokens)
        # Find start index of entity in the sentence
        start_idx = 0
        while start_idx <= len(sentence) - entity_length:
            if sentence[start_idx:start_idx + entity_length] == entity_tokens:

                # Check if these tokens are already labeled
                if all(idx not in labeled_positions for idx in range(start_idx, start_idx + entity_length)):
                    # Label these tokens
                    for i in range(entity_length):
                        labels[start_idx + i] = label
                        labeled_positions.add(start_idx + i)
                    break
            start_idx += 1
    return labels

def read_dataset(file_path):
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

def llm_predictions_to_bio(model_name, dataset_name, filename="predictions.txt", w_descriptions=False):
  if not w_descriptions:
    file_to_open = f"output/llm/predictions/{model_name}/{dataset_name}/{filename}"
  else:
    filename = filename.replace(".txt", "")
    filename += "_w_descriptions.txt"
    file_to_open = f"output/llm/predictions/{model_name}/{dataset_name}/{filename}"
  #lettura delle predizioni dal file e conversione
  with open(file_to_open, 'r') as f:
    lines = f.readlines()
    content = []
    i=1
    for line in lines:
      i=i+1
      entities = line.strip()
     
      #DEBUG 
      #print(i)
      try:
        content.append(ast.literal_eval(entities))
      except:
        #print("here")
        content.append([])
   
 
    # vengono rimosse le tuple che possono non essere conformi
    # ex. [("token"), ("token",""), ("","label"), ("token", "label", "label")]
 
    for i, tuples_list in enumerate(content):
      valid_tuples = []
      for t in tuples_list:
        if len(t) == 2:
          valid_tuples.append(t)
      content[i]=valid_tuples
 
    # si crea la lista di token predetti
    #sentences, gold_labels = read_dataset("Dataset/MedMentions-ZS/test.bio.txt")
    from datasets import load_dataset
    
    if dataset_name == "MedMentions-ZS":
      dataset = load_dataset("ibm/MedMentions-ZS", split="test")
    if dataset_name == "OntoNotes-ZS":
      dataset = load_dataset("alecocc/OntoNotes-ZS", split="test")
    if dataset_name == "LegalNER-ZS":
      dataset = load_dataset("alecocc/LegalNER-ZS", split="test")

    sentences = dataset['tokens']
    if dataset_name=="MedMentions-ZS":
      gold_labels = [[tag[:2] + medmentions_type_dict_inv()[tag[2:]].upper() if tag != "O" else tag for tag in tag_list] for tag_list in dataset['ner_tags']]
    else:
      gold_labels = [[tag[:2] + tag[2:] if tag != "O" else tag for tag in tag_list] for tag_list in dataset['ner_tags']]

    #print("GOLD LABELS:", gold_labels[0])
    predicted_labels = []
    for i, prediction in enumerate(content):
      predicted_labels.append(label_tokens(sentences[i], prediction))
    
 
    # si aggiungono i prefissi B- e I- del formato BIO
    for labels in predicted_labels:
      for i in reversed(range(len(labels))):
        if labels[i] is None:
          labels[i] = 'O'
        if labels[i] == '':
          labels[i] = 'O'
        if labels[i] != 'O':
          if i == 0 or labels[i] != labels[i-1]:
            labels[i] = 'B-' + labels[i]
          else:
            labels[i] = 'I-' + labels[i]
 
  return predicted_labels, gold_labels

# dataset_name = "OntoNotes-ZS"
# model_name = "Qwen2.5-7B-Instruct" #"Phi-3.5-mini-instruct" # meta-llama/Llama-3.1-8B-Instruct

parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]
datasets = args.datasets.split(",")
model_name = args.model_name.split("/")[-1]

w_descriptions = args.w_descriptions

for dataset_name  in datasets:
  preds, golds = llm_predictions_to_bio(model_name=model_name, dataset_name=dataset_name, w_descriptions=w_descriptions)
  #print(preds)

  f1_macro=f1_score(golds, preds, average="macro")
  recall_macro=recall_score(golds, preds, average="macro", zero_division=0)
  precision_macro=precision_score(golds, preds, average="macro", zero_division=0)

  f1_micro=f1_score(golds, preds, average="micro")
  recall_micro=recall_score(golds, preds, average="micro", zero_division=0)
  precision_micro=precision_score(golds, preds, average="micro", zero_division=0)

  report = classification_report(golds, preds, zero_division=0)
  print(report)

  print("F1 macro ", f1_macro)
  #print("F1 macro zero_division=1", f1_macro_2)
  os.makedirs(f"output/llm/results/{model_name}", exist_ok=True)
  with open(f"output/llm/results/{model_name}/res_{dataset_name}_2.jsonl", 'a') as f:
      result = {
        "model": model_name,
        "w_desc": w_descriptions,
        "recall_macro": recall_macro,
        "precision_macro": precision_macro,
        "f1_macro": f1_macro,
        "recall_micro": recall_micro,
        "precision_micro": precision_micro,
        "f1_micro": f1_micro
      }
      json.dump(result, f, ensure_ascii=False)
      f.write("\n")
    


# import json
# score = {}
# for model in models:
#   sum=0
#   score[model] = {}
#   for dataset in datasets:
#     f1_micro=f1_score(true[model][dataset], preds[model][dataset], average="micro")
#     score[model][dataset] = round(f1_micro*100,2)
#     sum = sum+score[model][dataset]
#   score[model]["AVG"] = round(sum/len(datasets),2)

# with open("llm_results_base.json", 'w') as json_file:
#     json.dump(score, json_file, indent=4)