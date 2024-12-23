"""Loads and manages entity descriptions"""
import os

from src.zeroner.utils.readers import read_entity_descriptions, get_zero_shot_splits



def get_entity_descriptions_negative(mode, dataset, split, filter_classes, entity_description_path, setting="zero-shot"):
    
    des = default_entity_descriptions_neg(os.path.join(entity_description_path + "/" + dataset + "/descriptions.txt"), dataset, filter_classes, setting)[split]

    return des['labels'], des['descriptions']


def default_entity_descriptions_neg(path, dataset, filter_classes, setting="zero-shot"):
    entity_descriptions = read_entity_descriptions(path)
    print("DATASET", dataset)
    train_types, dev_types, test_types = get_zero_shot_splits(dataset)
    print("TRAIN TYPES", train_types)
    print("DEV TYPES", dev_types)
    print("TEST TYPES", test_types)

    entity_descriptions_splits = {"train": {'labels': [], 'descriptions': []},
                                  'dev': {'labels': [], 'descriptions': []}, "test": {'labels': [], 'descriptions': []}}
    
    entity_descriptions_splits['train']['labels'].append('NEG')
    entity_descriptions_splits['train']['descriptions'].append(entity_descriptions['NEG'])
    entity_descriptions_splits['dev']['labels'].append('NEG')
    entity_descriptions_splits['dev']['descriptions'].append(entity_descriptions['NEG'])
    entity_descriptions_splits['test']['labels'].append('NEG')
    entity_descriptions_splits['test']['descriptions'].append(entity_descriptions['NEG'])

    for key, value in entity_descriptions.items():
        
        if setting == "zero-shot":
            if key in train_types:
                entity_descriptions_splits['train']['labels'].append(key)
                entity_descriptions_splits['train']['descriptions'].append(value)
            elif key in dev_types:
                entity_descriptions_splits['dev']['labels'].append(key)
                entity_descriptions_splits['dev']['descriptions'].append(value)
            elif key in test_types:
                entity_descriptions_splits['test']['labels'].append(key)
                entity_descriptions_splits['test']['descriptions'].append(value)
        
    return entity_descriptions_splits