"""Evaluation classes for tagging and classification"""

import logging
from textwrap import indent
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import classification_report
import seqeval.metrics

logger = logging.getLogger(__name__)


class evaluator_tagger(object):

    def __init__(self, args, labels, predictions, entity_labels, sep_indicies, texts = None, prediction_probs = None, description_types = None, save_results = False):
        self.args = args
        self.labels = labels
        self.predictions = predictions
        self.entity_labels = entity_labels
        self.sep_indicies = sep_indicies
        self.prediction_probs = prediction_probs
        self.description_types = description_types
        self.texts = texts
        self.scores = {}
        self.scores_chunk = {}
        self.scores_recognition = {}
        self.scores_recognition_chunk = {}
        self.save_results = save_results


        if 'multiclass' in args.mode:
            self.evaluate_transformer(args, labels, predictions, entity_labels, sep_indicies, texts)
        

    def flat_labels(self, labels, predictions):
        labels_flat = [item for sublist in labels for item in sublist]
        predictions_flat = [item for sublist in predictions for item in sublist]
        return labels_flat, predictions_flat

    def evaluate_transformer(self, args, labels, predictions, entity_labels, sep_indicies, texts):
        labels = [[tok for tok in sentence[:sep_indicies[i]]] for i, sentence in enumerate(labels)]
        predictions = [[tok for tok in sentence[:sep_indicies[i]]] for i, sentence in enumerate(predictions)]

        if texts != None:
            labels_no_sub = [[tok for j, tok in enumerate(sentence[:sep_indicies[i]]) if '#' not in texts[i][j]] for i, sentence in enumerate(labels)]
            predictions_no_sub = [[tok for j, tok in enumerate(sentence[:sep_indicies[i]]) if '#' not in texts[i][j]] for i, sentence in enumerate(predictions)]
        else:
            labels_no_sub = labels
            predictions_no_sub = predictions

        labels_iob, predictions_iob = self._convert_to_iob(labels_no_sub, predictions_no_sub, entity_labels)

        
        with open("./save_predictions_test.txt", "w+") as f:
            d = []
            for l, p, text in zip(labels_iob, predictions_iob, texts):
                d.append({"predictions": p, "labels": l, "text": text})
            import json
            json.dump(d, f, indent=4)

        acc = seqeval.metrics.accuracy_score(labels_iob, predictions_iob)
        f1_macro = seqeval.metrics.f1_score(labels_iob, predictions_iob, average='macro')
        f1_micro = seqeval.metrics.f1_score(labels_iob, predictions_iob, average='micro')
        prec_macro = seqeval.metrics.precision_score(labels_iob, predictions_iob, average='macro')
        prec_micro = seqeval.metrics.precision_score(labels_iob, predictions_iob, average='micro')
        recall_macro = seqeval.metrics.recall_score(labels_iob, predictions_iob, average='macro')
        recall_micro = seqeval.metrics.recall_score(labels_iob, predictions_iob, average='micro')

        self.scores_chunk['per_class'] = {}
        self.scores_chunk['overall'] = {}
        self.scores_chunk['overall']['acc'] = float(acc)
        self.scores_chunk['overall']['f1_macro'] = float(f1_macro)
        self.scores_chunk['overall']['f1_micro'] = float(f1_micro)
        self.scores_chunk['overall']['recall_macro'] = float(recall_macro)
        self.scores_chunk['overall']['recall_micro'] = float(recall_micro)
        self.scores_chunk['overall']['precision_macro'] = float(prec_macro)
        self.scores_chunk['overall']['precision_micro'] = float(prec_micro)

        class_report = seqeval.metrics.classification_report(labels_iob, predictions_iob)
        class_report_dict = seqeval.metrics.classification_report(labels_iob, predictions_iob, output_dict=True)
        for key, value in class_report_dict.items():
            if key not in ['micro avg', 'macro avg', 'weighted avg']:
                self.scores_chunk['per_class'][key] = {}
                for tag, score in value.items():
                    self.scores_chunk['per_class'][key][tag] = float(score)
        logger.info("######################### SPAN LEVEL scores ################################")
        logger.info(class_report)
        logger.info("acc: {}, rec_micro: {}, precision_micro: {}, f1_micro : {}, rec_macro: {}, precision_macro: {}, f1_macro : {}".format(acc, recall_micro, prec_micro, f1_micro, recall_macro, prec_macro, f1_macro))


        #logger.info("######################### EXCLUDE SUBTOKENS ################################")

        labels, predictions = self.flat_labels(labels_no_sub, predictions_no_sub)
        self.scores = self._calculate_scores(labels, predictions, entity_labels)

        return self.scores, self.scores_chunk, self.scores_recognition, self.scores_recognition_chunk


    def _calculate_scores(self, labels, predictions, entity_labels, recognition = False):
        scores = {}
        
        label_types = [entity_labels[x] for x in labels]
        prediction_types = [entity_labels[x] for x in predictions]

        # uncomment for score with negative class included
        # scores['acc_w_neg'] = accuracy_score(label_types, prediction_types)
        # scores['recall_micro_w_neg'] = recall_score(label_types, prediction_types, average='micro')
        # scores['precision_micro_w_neg'] =  precision_score(label_types, prediction_types, average='micro')
        # scores['f1_micro_w_neg'] = f1_score(label_types, prediction_types, average='micro')
        # scores['f1_macro_w_neg'] = f1_score(label_types, prediction_types, average='macro')
        
        entity_labels2 = [x for x in entity_labels if x != 'NEG']
        scores['per_class'] = {}
        scores['overall'] = {}
        scores['overall']['acc'] = float(accuracy_score([x for x in label_types if x != 'NEG'], [x for i, x in enumerate(prediction_types) if label_types[i] != 'NEG']))
        scores['overall']['recall_macro'] = float(recall_score(label_types, prediction_types, labels = entity_labels2, average='macro'))
        scores['overall']['recall_micro'] = float(recall_score(label_types, prediction_types, labels = entity_labels2, average='micro'))
        scores['overall']['precision_macro'] =  float(precision_score(label_types, prediction_types, labels = entity_labels2, average='macro'))
        scores['overall']['precision_micro'] =  float(precision_score(label_types, prediction_types, labels = entity_labels2, average='micro'))
        scores['overall']['f1_micro'] = float(f1_score(label_types, prediction_types, labels = entity_labels2, average='micro'))
        scores['overall']['f1_macro'] = float(f1_score(label_types, prediction_types, labels = entity_labels2, average='macro'))
       
        class_report_dict = seqeval.metrics.classification_report([[lb] if lb != 'NEG' else ['O'] for lb in label_types], [[pr] if pr != 'NEG' else ['O'] for pr in prediction_types], output_dict=True)
        class_report = seqeval.metrics.classification_report([[lb] if lb != 'NEG' else ['O'] for lb in label_types], [[pr] if pr != 'NEG' else ['O'] for pr in prediction_types])
        for key, value in class_report_dict.items():
            
            if key not in ['micro avg', 'macro avg', 'weighted avg']:
                scores['per_class'][key] = {}
                for tag, score in value.items():
                    
                    scores['per_class'][key][tag] = float(score)

        logger.info("########## TOKEN LEVEL scores ###############")
        
        logger.info(class_report)
        logger.info("######## Multi-class scores w/o neg class (O) #######")
        logger.info("acc: {}, rec_micro: {}, precision_micro: {}, f1_micro : {}, rec_macro: {}, precision_macro: {}, f1_macro : {}".format(scores['overall']['acc'], scores['overall']['recall_micro'], scores['overall']['precision_micro'], scores['overall']['f1_micro'], scores['overall']['recall_macro'], scores['overall']['precision_macro'], scores['overall']['f1_macro']))
        return scores


    def _convert_to_iob(self, labels_no_sub, predictions_no_sub, entity_labels, recognition=False):
        labels_iob = []
        predictions_iob = []


        for i in range(len(labels_no_sub)):
            labels_sent = []
            pred_sent = []
            for j in range(len(labels_no_sub[i])):
                curr_lab = labels_no_sub[i][j] - (1 if 'multiclass' not in self.args.mode else 0)
                lab = entity_labels[curr_lab] if not recognition else 'ENT'
                if labels_no_sub[i][j] != 0:
                    if j-1 < 0 or labels_no_sub[i][j -1] ==0:
                        labels_sent.append('B-' + lab)
                    else:# j+1 < len(labels_no_sub) and labels_no_sub[i][j +1] !=0:
                        labels_sent.append('I-' + lab)
                else:
                    labels_sent.append('O')

            for j in range(len(predictions_no_sub[i])):
                curr_lab = predictions_no_sub[i][j] - (1 if 'multiclass' not in self.args.mode else 0)
                lab = entity_labels[curr_lab] if not recognition else 'ENT'
                # if recognition:
                #     lab = 'ENT' if labels_no_sub[i][j] >= 1 else 'NEG'
                if predictions_no_sub[i][j] != 0:
                    if j-1 < 0 or predictions_no_sub[i][j -1] ==0:
                        pred_sent.append('B-' + lab)
                    else:# j+1 < len(labels_no_sub) and labels_no_sub[i][j +1] !=0:
                        pred_sent.append('I-' + lab)
                else:
                    pred_sent.append('O')
            labels_iob.append(labels_sent)
            predictions_iob.append(pred_sent)

            # print(labels_iob)

        return labels_iob, predictions_iob