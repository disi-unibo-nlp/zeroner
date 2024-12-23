""" Prepares batch data for different models"""
import gensim
import torch
import numpy as np


class BatchProcessor(object):

    def __init__(self, args, model, dataset):
        self.args = args
        self.model = model
        self.dataset = dataset
        self.model_type = args.model_type
        self.device = args.device

    def collator(self, data):
        batch = self.tagger_multiclass_collator(data)
        return batch

    def batch_to_dict(self, batch):
        
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "sep_index" : batch[3],
            "seq_mask" : batch[4],
            "split" : batch[5],
            "labels": batch[6],
        }

        return inputs

    
    def tagger_multiclass_collator(self, data):
            input_ids_lists = [f.input_ids_list for f in data]
            input_masks_lists = [f.input_mask_list for f in data]
            segment_ids_lists = [f.segment_ids_list for f in data]
            label_ids = [f.label_id for f in data]
            label_dict = [f.label_dict for f in data]
            batch_size = len(input_ids_lists)

            sep_index = [f.sep_index for f in data]
            max_sep_index = max(sep_index)

            #print(max_sep_index)

            padded_bert_tokens_list = []
            padded_bert_masks_list = []
            padded_bert_segments_list = []

            padded_bert_labels = []
            padded_sequence_mask = []

            sentence_lengths = [max([len(bert_tokens) for bert_tokens in input_ids_lists[i]]) for i in range(len(input_ids_lists))]
            longest_sent = max(sentence_lengths)

            for i in range(batch_size):
                padded_bert_tokens = []
                padded_bert_masks = []
                padded_bert_segments = []

                padding_label = [0] * (max_sep_index - len(label_ids[i]))
                for j in range(len(input_ids_lists[i])):
                    padding = [0] * (longest_sent - len(input_ids_lists[i][j]))
                    padded_bert_tokens.append(input_ids_lists[i][j] + padding)
                    padded_bert_masks.append(input_masks_lists[i][j] + padding)
                    padded_bert_segments.append(segment_ids_lists[i][j] + padding)

                bert_tokens_t = torch.tensor(padded_bert_tokens).to(device=self.device)
                bert_masks_t = torch.tensor(padded_bert_masks).to(device=self.device)
                bert_segments_t = torch.tensor(padded_bert_segments).to(device=self.device)

                padded_bert_tokens_list.append(bert_tokens_t)
                padded_bert_masks_list.append(bert_masks_t)
                padded_bert_segments_list.append(bert_segments_t)

                padded_bert_labels.append(label_ids[i] + padding_label) #MODIFY BACK
                padded_sequence_mask.append(sep_index[i] * [1] + (max_sep_index - sep_index[i]) *[0])

            bert_tokens_t = torch.stack(padded_bert_tokens_list).transpose(1,0)
            bert_masks_t = torch.stack(padded_bert_masks_list).transpose(1,0)
            bert_segments_t = torch.stack(padded_bert_segments_list).transpose(1,0)
            bert_labels_t = torch.tensor(padded_bert_labels).to(device=self.device, dtype=torch.long)
            bert_seq_mask_t = torch.tensor(padded_sequence_mask).to(device=self.device,dtype=torch.uint8)
            bert_sep_t = torch.tensor(sep_index).to(device=self.device)

            if self.dataset.split == 'train':
                split = 0
            elif self.dataset.split == 'dev':
                split = 1
            else:
                split = 2
            split = torch.tensor(split).to(device=self.device)

            return bert_tokens_t, bert_masks_t, bert_segments_t, bert_sep_t, bert_seq_mask_t, split, bert_labels_t