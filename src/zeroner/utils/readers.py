import os

import spacy
from nltk.tokenize import sent_tokenize

from src.zeroner.utils.wrappers import Token
ontonotes_splits = {
    "train_types": ['ORG', 'GPE', 'DATE', 'PERSON'],
    "dev_types": ['EVENT', 'PRODUCT', 'LAW', 'NORP'],
    "test_types": ['LOC', 'FAC', 'WORK_OF_ART'] #'TIME', 'QUANTITY', 'LANGUAGE', 'CARDINAL',
} 

medmentions_splits = {
    "train_types": ['Biologic_Function', 'Chemical', 'Health_Care_Activity', 'Anotomical_Structure', "Finding", "Spatial_Concept", 
                     "Intellectual_Product", "Research_Activity", 'Medical_Device', 'Eukaryote', 'Population_Group'],
    "dev_types": ['Biomedical_Occupation_or_Discipline', 'Virus', 'Clinical_Attribute', 'Injury_or_Poisoning', 'Organization'],
    "test_types": ['Body_System', 'Food', 'Body_Substance', 'Bacterium', 'Professional_or_Occupational_Group']
}

legalner_splits = {
    "train_types": ['COURT', 'PETITIONER', 'RESPONDENT', 'LAWYER', 'PROVISION', 'OTHER_PERSON'],
    "dev_types": ['JUDGE', 'ORG', 'STATUTE'],
    "test_types": ['GPE', 'PRECEDENT', 'CASE_NUMBER', 'WITNESS']
}


def get_zero_shot_splits(dataset):
    if dataset == "medmentions" or dataset == "MedMentions-ZS":
        return medmentions_splits["train_types"], medmentions_splits["dev_types"], medmentions_splits["test_types"]
    elif dataset == "ontonotes" or dataset == "OntoNotes-ZS":
        return ontonotes_splits["train_types"], ontonotes_splits["dev_types"], ontonotes_splits["test_types"]
    elif dataset == "legalner" or dataset == "LegalNER-ZS":
        return legalner_splits['train_types'], legalner_splits['dev_types'], legalner_splits['test_types']	


def medmentions_type_dict_inv():
    return {'T058': "Health_Care_Activity", "T062": "Research_Activity", "T037": "Injury_or_Poisoning",
            "T038": "Biologic_Function", "T005": "Virus", "T007": "Bacterium", "T204": "Eukaryote",
            "T017": "Anotomical_Structure", "T074": "Medical_Device", "T031": "Body_Substance", "T103": "Chemical",
            "T168": "Food", "T201": "Clinical_Attribute", "T033": "Finding", "T082": "Spatial_Concept",
            "T022": "Body_System", "T091": "Biomedical_Occupation_or_Discipline", "T092": "Organization",
            "T097": "Professional_or_Occupational_Group", "T098": "Population_Group", "T170": "Intellectual_Product",
            "NEG": "NEG"}


def medmentions_type_dict():
    return {v: k for k, v in medmentions_type_dict_inv().items()}


def read_data_iob(input_path, split, id=False, limit=None):
    with open(input_path+ "/" + split + ".bio.txt", "r") as f:
        lines = f.readlines()
    sentences = []
    labels = []
    sentence = []
    label = []
    tag2id = {"O": 0}
    start_id = 1
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
            if tag != 'O' and tag not in tag2id:
                tag2id[f"B-{tag.split('-')[1]}"] = start_id
                tag2id[f"I-{tag.split('-')[1]}"] = start_id + 1
                start_id += 2

    if sentence:
        sentences.append(sentence)
        labels.append(label)

    
    content = []
    for tokens, tags in zip(sentences, labels):
        final_sentence = []
        for i in range(len(tokens)):
            token = Token(word=tokens[i], type=tags[i], type_id=tag2id[tags[i]])
            final_sentence.append(token)
        content.append(final_sentence)

    return content 

def read_entity_descriptions(path):
    nlp = spacy.load("en_core_web_sm")
    descriptions = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            content = line.split('\t')
            print("Content:", content)
            sent = sent_tokenize(content[1].strip())
            tokenized = " ".join([' '.join([el.text for el in nlp(sen)]) for sen in sent])
            descriptions[content[0]] = tokenized
    return descriptions
