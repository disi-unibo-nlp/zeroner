"""Testing pipeline"""
import argparse
import logging
import os
import sys
import nltk
nltk.download('punkt')
import torch
from src.zeroner.models.trainer import Trainer

from src.zeroner.dataloaders.dataloader import EntityRecognitionMultiClassDataset
from src.zeroner.models.entity_descriptions import get_entity_descriptions_negative
from src.zeroner.utils.evaluation import evaluator_tagger
from src.zeroner.utils.readers import  read_data_iob #, read_data_legalner

logger = logging.getLogger(__name__)


def run_test(args):
   
    logger.info("READ DATA: {}".format(args.dataset))
   
    data = read_data_iob(args.input_path, args.split)
    logger.info(f"DATA LEN: {data[0]}")

    # Loads the descriptions inclusive negative description if we are in tagging and multiclass mode
    logger.info("USING ENTITY DESCRIPTION: {}".format(args.entity_descriptions_mode))
    if 'tagger' in args.mode and (
            ('multiclass' in args.mode and 'transformer' == args.model) or 'unsupervised_baseline' == args.model):
        entity_labels, entity_descriptions = get_entity_descriptions_negative(args.entity_descriptions_mode,
                                                                              args.dataset, args.split,
                                                                              filter_classes='filtered_classes' in args.mode,
                                                                              entity_description_path=args.entity_description_path)
        logger.info(f"ENTITIES: {entity_labels}")
        logger.info(f"DESCS: {entity_descriptions}")
   
    if args.model == 'transformer':
        if args.mode in ['tagger_multiclass', 'tagger_multiclass_filtered_classes']:
            dataloader = EntityRecognitionMultiClassDataset(args, args.split, data, entity_labels, entity_descriptions, remove_negatives=True)

        trainer = Trainer(args, dataloader)
        labels, preds, preds_prob, _ = trainer.predict()

        # Run the evaluator for the respective task
        if 'tagger' in args.mode:
            entity_labels_dev = dataloader.entity_labels
            sep_indicies = [f.sep_index for f in dataloader.features]
            if 'multiclass' in args.mode:
                texts = [f.text_list[0] for f in dataloader.features] if args.model == 'transformer' else None
                evaluator = evaluator_tagger(args, labels, preds, entity_labels_dev, sep_indicies, texts,
                                             save_results=True)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Args for evaluation")
    parser.add_argument('--split', choices=['train', 'dev', 'test', "conll-2012-test"], type=str)
    parser.add_argument('--mode', type=str, help="Select mode")
    parser.add_argument('--model', type=str, help="Select model")
    parser.add_argument('--model_id', type=str, help="Select model id from HuggingFace repo", default="bert-large-cased")

    parser.add_argument('--dataset', type=str, help="Select statistic")
    parser.add_argument('--input_path', default='./data/OntoNotes_5.0_NER_BIO/conll-ontonotes-paper-wo-nt-zero-shot',
                        help="Input path")
    parser.add_argument('--entity_descriptions_mode', default='wordnet', help="Input path")
    parser.add_argument("--overwrite_dumped", action="store_true",
                        help="Mask the entity in the text to assist generalization")
    parser.add_argument('--symbols_file', type=str, help="Specify symbols file", default="symbols01")
    parser.add_argument('--max_symbol_length', type=int, default=5)

    parser.add_argument("--output_dir", default="", type=str, help="Pretrained name or path of existing model")
    parser.add_argument("--checkpoint", default=None, type=str, help="The checkpoint from which to continue training.")

    parser.add_argument("--add_iob", action="store_true", help="Mask the entity in the text to assist generalization")
    parser.add_argument("--only_keep_relevant_classes", action="store_true",
                        help="Mask the entity in the text to assist generalization")

    parser.add_argument("--max_sequence_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences")
    parser.add_argument("--max_description_length", default=100, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences")
    parser.add_argument("--add_negative_samples", action="store_true",
                        help="Mask the entity in the text to assist generalization")
    parser.add_argument("--mask_entity_partially", action="store_true",
                        help="Mask the entity in the text to assist generalization")
    parser.add_argument("--mask_entity", action="store_true",
                        help="Mask the entity in the text to assist generalization")
    parser.add_argument("--negative_samples_ratio", type=float, default=0.5,
                        help="Mask the entity in the text to assist generalization")
    parser.add_argument("--mask_probability", default=0.5, type=float, help="Dropout of linear layer.")

    parser.add_argument('--model_type', type=str, help="Select model")
    parser.add_argument('--use_training_tfidf', action='store_true',
                        help="Use training unsupervised for tf-idf measure")
    parser.add_argument("--linear_units_symbol", default=100, type=int, help="Dropout of linear layer.")
    parser.add_argument('--embedding_path', type=str, help="Specify embedding path", default="../data/cc.en.300.vec")
    parser.add_argument("--add_boundaries", action="store_true",
                        help="Mask the entity in the text to assist generalization")
    parser.add_argument("--linear_dropout", default=0.2, type=float, help="Dropout of linear layer.")
    parser.add_argument('--config_path', default='configs/bert-base-default.json', help="Input path of config file")
    parser.add_argument('--limit', type=int, help="Input path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name", )
    parser.add_argument("--per_gpu_eval_batch_size", default=100, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--entity_description_path', type=str, help="Entity description path",
                        default="./xencoder/data/entity_descriptions")
    # For tf-idf
    parser.add_argument("--wikipedia_path", type=str, help="path to wikipedia dump", )

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")

    args = parser.parse_args()

    if args.model == 'transformer' and not os.path.exists(args.output_dir):
        print('Output path does not exist. Cannot load model. Abort...')
        sys.exit()
    logging.basicConfig(format='%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s',
                        datefmt='%H:%M:%S', level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])

    args.device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    run_test(args)
