""" Transformers for zero-shot NER"""
from bdb import set_trace
import os
import random

import torch
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_bert_model(args, num_classes=None):
    if os.path.isfile(os.path.join(args.output_dir, args.checkpoint, "pytorch_model.bin")):
        print("Load model from:", os.path.join(args.output_dir, args.checkpoint, "pytorch_model.bin"))
    
    if args.model_type == "BertTaggerMultiClass":
        return BertTaggerMultiClass.from_pretrained(os.path.join(args.output_dir, args.checkpoint),
                                                    output_hidden_states=True).to(device) if os.path.isfile(
            os.path.join(args.output_dir, args.checkpoint,
                         "pytorch_model.bin")) else BertTaggerMultiClass.from_pretrained(args.model_id, #'dmis-lab/biobert-base-cased-v1.1',
                                                                                         architectures=[
                                                                                             "BertTaggerMultiClass"],
                                                                                         output_hidden_states=True,
                                                                                         finetuning_task={
                                                                                             "dropout_prob": args.linear_dropout,
                                                                                             "transformer_finetune": True,
                                                                                             "num_labels": num_classes,
                                                                                             "no_cuda": args.no_cuda,
                                                                                             "description_mode": args.entity_descriptions_mode,
                                                                                             "dataset": args.dataset,
                                                                                             "filter_classes": 'filtered_classes' in args.mode,
                                                                                             "dataset": args.dataset}).to(device)
     





class BertTaggerMultiClass(BertPreTrainedModel):
    '''
    Use plain Bert Model for Textual Inference
    see https://github.com/huggingface/pytorch-pretrained-BERT/blob/ee0308f79ded65dac82c53dfb03e9ff7f06aeee4/pytorch_pretrained_bert/modeling.py#L938
    Small changes to the model described in the paper: max pooling over both class representations and negative representation from independent encoding -- resulted in slightly better results.
    '''

    def __init__(self, config):

        super().__init__(config)

        self.bert = BertModel(
            config)  # from_pretrained(config.finetuning_task['model_name'], output_hidden_states = config.output_hidden_states, output_attentions = config.output_attentions)
        self.num_labels = config.finetuning_task['num_labels']  # 11#4#11#4#11#11
        self.drop = torch.nn.Dropout(config.finetuning_task['dropout_prob'])
        self.bert_output_size = config.hidden_size
        self.linear = torch.nn.Linear(self.bert_output_size, 1)  # Number of classes 2
        #self.linear_zero = torch.nn.Linear(self.num_labels * self.bert_output_size, 1)  # Number of classes 2
        self.linear_zero2 = torch.nn.Linear(self.bert_output_size, 1)  # Number of classes 2
        self.linear_zero3 = torch.nn.Linear(self.bert_output_size, 1)  # Number of classes 2
        self.softmax = torch.nn.LogSoftmax(dim=2)
        self.finetune = config.finetuning_task['transformer_finetune']
        self.use_symbols = False
        self.dataset = config.finetuning_task['dataset']
        #self.class_weight = [0.1 if self.dataset == 'medmentions' else 0.01] + [1] * (
        #        self.num_labels - 1)  # 0.01, 1 1 1 1 #0.1 for medmentions, ontonotes #FOR MEDMENTIONS use 0.1
        #self.class_weight = torch.FloatTensor(self.class_weight).to(device)
        # self.device = 'cpu' if config.finetuning_task['no_cuda'] else 'cuda:0'
        self.description_mode = config.finetuning_task['description_mode']
        self.dataset = config.finetuning_task['dataset']
        self.filter_classes = config.finetuning_task['filter_classes']

        self.init_weights()

    def forward(self, *args, input_ids, attention_mask, token_type_ids, sep_index, seq_mask, split, labels=None,
                **kwargs):
        # seq_transformer = batch_size x max_seq_length (padded) : sentence
        sep_index_max = torch.max(sep_index)
        predictions = []
        predictions_zero = []
        predictions_zero_base = []

        from transformers import BertTokenizer

        # Load the pre-trained BERT tokenizer (cased)
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        # Convert token IDs back to tokens
        
        # with open("prova_out2.txt", 'a') as f:
        #     f.write(f"***************************************************************\n")
        #     f.write(f"sep index: {sep_index}\n")
        #     f.write(f"input ids size: {input_ids.size()}\n")
        #     f.write(f"LEN input ids: {len(input_ids)}\n")
        #     f.write(f"sep_index size: {sep_index.size(0)}\n")
        #     f.write(f"sep index max: {sep_index_max.item()}")
        for j in range(input_ids.size(0)):
            if j == 0:
                # with open("prova_out2.txt", 'a') as f:
                #     tokens = tokenizer.convert_ids_to_tokens(input_ids[0][0])  # Convert tensor to list
                #     f.write(str(tokens) + '\n')  # Write the tokens to the file
                #     f.write(f"\n#### STOP 1  ####\n")
                #     token_ids_slice = input_ids[0][0, :sep_index_max.item()]  # Convert tensor slice to list
                #     tokens = tokenizer.convert_ids_to_tokens(token_ids_slice)  # Convert token IDs to tokens
                #     f.write("i: 1")
                #     f.write(str(tokens) + "\n")  # Write the tokens to the file
                #     token_ids_slice = input_ids[0][1, :sep_index_max.item()]
                #     tokens = tokenizer.convert_ids_to_tokens(token_ids_slice)  # Convert token IDs to tokens
                #     f.write("i: 2")
                #     f.write(str(tokens) + "\n")
                #     f.write(f"\n#### STOP 2####\n")
                inp_zero = torch.stack([input_ids[j][i, :sep_index_max.item()] for i in range(sep_index.size(0))])
                att_zero = torch.stack([attention_mask[j][i, :sep_index_max.item()] for i in range(sep_index.size(0))])
                tok_type_zero = torch.stack(
                    [token_type_ids[j][i, :sep_index_max.item()] for i in range(sep_index.size(0))])
                if not self.finetune:
                    with torch.no_grad():
                        words_out = \
                            self.bert(input_ids=inp_zero, attention_mask=att_zero, token_type_ids=tok_type_zero)[0]
                else:
                    words_out = self.bert(input_ids=inp_zero, attention_mask=att_zero, token_type_ids=tok_type_zero)[0]

                pooled_out = self.drop(words_out)
                logits = self.linear_zero3(pooled_out)
                predictions_zero_base.append(logits)
            else:
                if not self.finetune:
                    with torch.no_grad():
                        words_out = self.bert(input_ids=input_ids[j], attention_mask=attention_mask[j],
                                              token_type_ids=token_type_ids[j])[0]
                else:
                    words_out = self.bert(input_ids=input_ids[j], attention_mask=attention_mask[j],
                                          token_type_ids=token_type_ids[j])[0]  # res, layer_out

                words_out = torch.stack([words_out[i, :sep_index_max.item(), :] for i in range(words_out.size(0))])
                pooled_out = self.drop(words_out)
                predictions_zero.append(self.linear_zero2(pooled_out))
                logits = self.linear(pooled_out)
                predictions.append(logits)

        random.shuffle(predictions_zero)
        predictions_zero = torch.stack(predictions_zero_base + predictions_zero)  # .transpose(0,1).transpose(1,2)
        predictions_zero = predictions_zero.transpose(0, 1).transpose(1, 2)
        predictions_zero = predictions_zero.contiguous().view(predictions_zero.size(0), predictions_zero.size(1), -1)

        predictions_zero = torch.max(predictions_zero, dim=2)[0].unsqueeze(2)

        predictions = torch.stack(predictions)
        predictions = predictions.transpose(0, 1).transpose(1, 2).squeeze(3)

        # logits = self.softmax(torch.cat((predictions_zero, predictions), dim =2))
        logits = torch.cat((predictions_zero, predictions), dim=2)

        loss = 0

        if labels is not None:
            labels = torch.stack([labels[i, :sep_index_max.item()] for i in range(labels.size(0))])
            neg_weight = (labels != 0).sum()/((labels != 0).sum()+(labels == 0).sum())
            # with open("prova_out2.txt", 'a') as f:
            #     f.write(f"Neg weights: {neg_weight}")
            #     f.write("\n")
            class_weight = [neg_weight] + [1.0]*(logits.size(2)-1)
            class_weight = torch.FloatTensor(class_weight).to(device)
            weights = class_weight if split.item() == 0 else torch.FloatTensor([1] * logits.size(2)).to(device)
            loss_fct = CrossEntropyLoss(weight=weights)
            # loss_fct = CrossEntropyLoss()
            active_logits = logits.view(-1, logits.size(2))
            active_labels = labels.view(-1)
            loss = loss_fct(active_logits, active_labels)

        return logits, loss