""" Prepares the data given in BIO format for respective models and the respective task, i.e. transformer or LSTM"""
import itertools
import logging
import math
import os
import pickle
import sys
from collections import Counter
from random import random

from torch import is_tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer

logger = logging.getLogger(__name__)


class InputFeaturesMultiClass(object):
    """A single set of features of data."""

    def __init__(self, text_list, label_id, labels, label_dict=None, input_ids_list=None, input_mask_list=None,
                 segment_ids_list=None, sep_index=None, entity_index=None, tagging=None):
        self.text_list = text_list
        self.input_ids_list = input_ids_list
        self.input_mask_list = input_mask_list
        self.segment_ids_list = segment_ids_list
        self.label_id = label_id
        self.labels = labels
        self.label_dict = label_dict
        self.sep_index = sep_index
        self.tagging = tagging
        self.entity_index = entity_index

    def __str__(self):
        rep = ""
        if self.tagging:
            for i in range(len(self.text_list)):
                rep += '\n'.join(["Text: " + ' '.join(self.text_list[i]),
                                  "input_ids: " + " ".join([str(x) for x in self.input_ids_list[i]]),
                                  "input_mask: " + " ".join([str(x) for x in self.input_mask_list[i]]),
                                  "segment_ids: " + " ".join([str(x) for x in self.segment_ids_list[i]])])
            rep += '\n'
            rep += '\n'.join(["label_id: " + " ".join(str(x) for x in self.label_id),
                              "labels: " + " ".join(str(x) for x in self.labels),
                              'label_dict: ' + ','.join(str(x) + ":" + str(y) for x, y in self.label_dict.items()),
                              str("Sep index: " + str(self.sep_index))])
        else:
            for i in range(len(self.text_list)):
                rep += '\n'.join(["Text: " + ' '.join(self.text_list[i]),
                                  "input_ids: " + " ".join([str(x) for x in self.input_ids_list[i]]),
                                  "input_mask: " + " ".join([str(x) for x in self.input_mask_list[i]]),
                                  "segment_ids: " + " ".join([str(x) for x in self.segment_ids_list[i]])])
            rep += '\n'
            rep += '\n'.join(["label_id: " + str(self.label_id), str("Sep index: " + str(self.sep_index)),
                              "entity_index: " + str(self.entity_index[0]) + "," + str(self.entity_index[1])])
        return rep

    def __repr__(self):
        return " ".join(self.text_list[0])


class EntityRecognitionMultiClassDataset(Dataset):

    def __init__(self, args, split, sentences, entity_labels, entity_descriptions, vocabulary=None, vocabulary_inv=None,
                 all_labels=None, remove_negatives=True):
        # self.tokenizer_name = 'bert-large-uncased'
        # self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)
        self.sentences = sentences
        self.entity_labels = entity_labels
        self.features = []
        self.limit = args.limit
        self.max_seq_length = args.max_sequence_length
        self.dataset = args.dataset
        self.mode = args.mode
        self.model = args.model
        self.split = split
        self.output_dir = args.output_dir
        self.only_keep_relevant_classes = args.only_keep_relevant_classes
        self.mask_probability = 1 - args.mask_probability  # 0.3
        self.remove_negatives = remove_negatives

        if args.model == 'transformer':
            self.tokenizer_name = 'bert-large-cased'
            self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)
            self.mask_entity_partially = args.mask_entity_partially
            self.entity_descriptions = entity_descriptions
            self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)
            self.max_description_length = args.max_description_length
            self.mask_entity_partially = args.mask_entity_partially
            self.description_mode = args.entity_descriptions_mode
            self.features = self.process_text(sentences, entity_labels, entity_descriptions)

        else:
            logger.error("Error. No suitable mode found. Aborting...")
            sys.exit()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()
        return self.features[idx]

    def get_name(self):
        if self.model == 'transformer':
            return "_".join(
                [self.dataset, self.split, self.mode, self.model, self.tokenizer_name, self.description_mode,
                 str(self.max_description_length), str(self.max_seq_length), str(self.mask_entity_partially),
                 str(self.mask_probability), str(self.only_keep_relevant_classes)])

    def process_text(self, sentences, entity_labels, entity_descriptions):
        logger.info("Dataloader file {}".format(self.get_name()))
        # Currently ignoring sentences with no named entity of type
        if False:  # not self.limit and os.path.isfile(os.path.join('/tmp', self.get_name())):
            data_file = open(os.path.join('/tmp', self.get_name()), 'rb')
            features = pickle.load(data_file)
        else:
            print("LEN SENTENCES: ", len(sentences))
            print("LABELS: ", entity_labels)
            print("DECRIPTION ", entity_descriptions)
            features = []
            for i, sentence in enumerate(tqdm(sentences[:self.limit], desc="Sentence loop")):
                tokenized_text_list = []
                segment_ids_list = []
                input_ids_list = []
                input_mask_list = []
                break_it = False
                do_mask = random()
                mask_prob = self.mask_probability
                for k, description in enumerate(entity_descriptions):
                    premise = self.tokenizer.tokenize(' '.join(w.word for w in sentence))
                    premise = []
                    text_labels = []
                    text_labels_b = []
                    for word in sentence:
                        tok = self.tokenizer.tokenize(word.word)
                        premise += ["[unused1]"] * len(tok) if (
                                    len(word.type.split('-')) > 1 and word.type.split('-')[1] in entity_labels[
                                k] and self.mask_entity_partially and do_mask > mask_prob and self.split == 'train') else tok
                        text_labels += [word.type] * len(tok) if len(word.type.split('-')) == 1 or word.type.split('-')[
                            1] in entity_labels else ["O"] * len(tok)
                        text_labels_b += [word.type.split('B-')[1]] * len(tok) if len(word.type.split('B-')) > 1 and \
                                                                                  word.type.split('B-')[
                                                                                      1] in entity_labels else [
                                                                                                                   "NEG"] * len(
                            tok)

                    if 'B-' + entity_labels[k] not in text_labels and self.only_keep_relevant_classes:
                        continue

                    if self.remove_negatives and set(text_labels) == set(["O"]):
                        break_it = True
                        break
                    # text_labels = [label.split("-")[1]  if len(label.split("-")) > 1 else label for label in text_labels] if not self.add_iob else text_labels
                    hypothesis = self.tokenizer.tokenize(description)
                    diff_descr = len(hypothesis) - self.max_description_length

                    if diff_descr < 0:
                        diff_descr = len(hypothesis)

                    tokenized_text = ["[CLS]"] + premise + ["[SEP]"] + hypothesis[:diff_descr] + ["[SEP]"]
                    ind_split = tokenized_text.index("[SEP]")
                    remainder = len(tokenized_text) - ind_split
                    max_input = self.max_seq_length - (remainder + 1)
                    samples_num = math.ceil(ind_split / max_input)

                    ind = [0]
                    if samples_num > 1:
                        ind.append(max_input)
                    else:
                        ind.append(ind_split)

                    for i in range(len(ind) - 1):
                        tokenized_text_f = tokenized_text[ind[i]:ind[i + 1]] + tokenized_text[ind_split:]
                        segment_ids = ([0] * (ind[i + 1] - ind[i])) + ([1] * remainder)
                        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text_f)
                        # input_mask = ([1] * ind_split) + ([0] * (len(input_ids) - ind_split))

                        input_mask = [1] * len(input_ids)
                        assert (len(input_ids) == len(segment_ids))
                        assert (len(input_mask) == len(input_ids))

                        tokenized_text_list.append(tokenized_text_f)
                        segment_ids_list.append(segment_ids)
                        input_ids_list.append(input_ids)
                        input_mask_list.append(input_mask)

                if self.remove_negatives and break_it:
                    continue

                text_labels = ["NEG"]
                for word in sentence:
                    tok = self.tokenizer.tokenize(word.word)
                    text_labels += len(tok) * [word.type.split('-')[1]] if len(word.type.split('-')) > 1 and \
                                                                           word.type.split('-')[
                                                                               1] in entity_labels else len(tok) * [
                        'NEG']

                text_labels_ids = [entity_labels.index(label) for label in text_labels]
                label_dict = {x: entity_labels[x] for x in set(text_labels_ids)}
                text_labels_ids = text_labels_ids  # CLS Token

                sep_index = tokenized_text_list[0].index("[SEP]")
                features.append(InputFeaturesMultiClass(text_list=tokenized_text_list, input_ids_list=input_ids_list,
                                                        input_mask_list=input_mask_list,
                                                        segment_ids_list=segment_ids_list,
                                                        sep_index=sep_index,
                                                        label_dict=label_dict,
                                                        tagging=True,
                                                        label_id=text_labels_ids[:sep_index],
                                                        labels=text_labels[:sep_index]))

            if not self.limit:
                data_file = open(os.path.join('/tmp', self.get_name()), 'wb')
                pickle.dump(features, data_file)

        #features = [features[0], features[1], features[2], features[4], features[8]]

        logger.info("Number samples: {}".format(len(features)))
        logger.info("Example features: {}".format(features[0]))
        print("Number samples: {}".format(len(features)))
        print("Example features: {}".format(features[0]))

        with open(os.path.join(self.output_dir, "processed_text_" + self.split + ".txt"), "w") as f:
            for element in features[:100]:
                f.write("{}\n".format(element))
        return features


