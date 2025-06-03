"""
Named Entity Recognition Evaluation Script

This script evaluates NER models using the zshot library on multiple datasets.
Supports both GLINER and SMXM linkers with configurable parameters.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Union
from dotenv import load_dotenv
load_dotenv()

import logging
from datetime import datetime
import numpy as np
import spacy
from zshot import PipelineConfig
from zshot.linker import LinkerSMXM, LinkerGLINER
from zshot.utils.data_models import Entity
from zshot.evaluation.zshot_evaluate import evaluate
from zshot.evaluation.metrics._seqeval._seqeval import Seqeval

HF_TOKEN = os.getenv('HF_TOKEN', None)
if HF_TOKEN is None:
    raise ValueError("HuggingFace token not found. Please set the HF_TOKEN environment variable.")
else:
    from huggingface_hub import login
    login(HF_TOKEN)

# Step 1: Early basic setup — logs to console
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S"
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  

id2tag = {'T058': "Health_Care_Activity", "T062": "Research_Activity", "T037": "Injury_or_Poisoning",
            "T038": "Biologic_Function", "T005": "Virus", "T007": "Bacterium", "T204": "Eukaryote",
            "T017": "Anotomical_Structure", "T074": "Medical_Device", "T031": "Body_Substance", "T103": "Chemical",
            "T168": "Food", "T201": "Clinical_Attribute", "T033": "Finding", "T082": "Spatial_Concept",
            "T022": "Body_System", "T091": "Biomedical_Occupation_or_Discipline", "T092": "Organization",
            "T097": "Professional_or_Occupational_Group", "T098": "Population_Group", "T170": "Intellectual_Product",
            "NEG": "NEG"}


class NEREvaluator:
    """Named Entity Recognition evaluator for zshot models."""
    
    def __init__(self, model_hf: str, dataset: str, split: str, 
                 type_case: str, device: str = 'cuda'):
        """
        Initialize the NER evaluator.
        
        Args:
            model_hf: HuggingFace model name
            dataset: Dataset name (e.g., 'ontonotes', 'medmentions', 'legalner', 'all')
            split: Dataset split (e.g., 'validation')
            type_case: Case type for entities (e.g., 'lowercase') - only used for GLiNER models
            device: Device to run the model on
        """
        self.model_hf = model_hf
        self.model_name = model_hf.split("/")[-1]
        self.dataset = dataset
        self.split = split
        self.type_case = type_case
        self.device = device
        self.is_gliner = "gliner" in self.model_hf.lower()
        
        # Define available datasets
        self.available_datasets = ["ontonotes", "medmentions", "legalner"]
        
    def get_datasets_to_evaluate(self) -> List[str]:
        """Get list of datasets to evaluate based on user input."""
        if self.dataset == "all":
            return self.available_datasets
        else:
            return [self.dataset]
    
    def load_entity_mapping(self, mapping_file: str, dataset_name: str) -> Dict[str, Any]:
        """Load entity type mapping from JSON file for a specific dataset."""
        try:
            with open(mapping_file, 'r') as f:
                entity_mappings = json.load(f)
                
            if self.is_gliner:
                entity_mapping = entity_mappings[dataset_name][self.split][self.type_case]
            else:
                entity_mapping = entity_mappings[dataset_name][self.split]
                
            logger.info(f"Entity mapping loaded for {dataset_name}: {entity_mapping}")
            logger.info(f"Entities: {list(entity_mapping.values() if self.is_gliner else entity_mapping.keys())}")
            return entity_mapping
            
        except (FileNotFoundError, KeyError) as e:
            raise ValueError(f"Failed to load entity mapping for {dataset_name}: {e}")
    
    def setup_pipeline(self, entity_mapping: Dict[str, Any]) -> None:
        """Set up the spaCy pipeline with zshot for a specific entity mapping."""
        if self.is_gliner:
            entities = [Entity(name=ent_name) for ent_name in entity_mapping.values()]
        else:
            entities = [Entity(name=ent_name, description=ent_descr) 
                       for ent_name, ent_descr in entity_mapping.items()]
        
        logger.info(f"Entities for {self.model_name}:")
        for ent in entities:
            logger.info(f"  - {ent.name}")
            if hasattr(ent, 'description') and ent.description:
                logger.info(f"    Description: {ent.description}")
        
        # Choose linker based on model name
        if self.is_gliner:
            linker = LinkerGLINER(model_name=self.model_hf)
        else:
            linker = LinkerSMXM(model_name=self.model_hf)
        
        # Initialize spaCy pipeline
        nlp = spacy.blank("en")
        nlp_config = PipelineConfig(
            linker=linker,
            entities=entities,
            device=self.device
        )
        nlp.add_pipe("zshot", config=nlp_config, last=True)
        return nlp
    
    def convert_to_tag(self, example: Dict, entity_mapping: Dict[str, str]) -> Dict:
        """Convert NER tags using entity mapping for GLiNER models."""
        tokens = example['tokens']
        ner_tags = example['ner_tags']
        converted_tags = [
            (tag[:2] + entity_mapping[tag[2:]]).replace("_", " ") 
            if tag != 'O' else 'O' 
            for tag in ner_tags
        ]
        return {'tokens': tokens, 'ner_tags': converted_tags}
    
    def convert_medmentions(self, example):
        tokens = example['tokens']
        ner_tags = example['ner_tags']
        example = [ (x[:2] + id2tag[x[2:]]).upper() if x != 'O' else 'O' for x in ner_tags]
        return {'tokens': tokens, 'ner_tags': example}
    
    def load_and_process_dataset(self, dataset_name: str, entity_mapping: Dict[str, Any]):
        """Load and process a specific dataset."""
        logger.info(f"Loading {dataset_name} dataset for {self.model_name}")

        if dataset_name == "ontonotes":
            from zshot.evaluation.dataset import load_ontonotes_zs
            dataset_zs = load_ontonotes_zs(self.split)
        elif dataset_name == "medmentions":
            from zshot.evaluation.dataset import load_medmentions_zs
            dataset_zs = load_medmentions_zs(self.split)
            dataset_zs = dataset_zs.map(self.convert_medmentions)
        elif dataset_name == "legalner":
            from datasets import load_dataset
            dataset_zs = load_dataset("alecocc/LegalNER-ZS", split=self.split)
            dataset_zs = dataset_zs.filter(lambda x: "B-" in " ".join(x['ner_tags']))
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        if self.is_gliner:
            processed_dataset = dataset_zs.map(
                lambda example: self.convert_to_tag(example, entity_mapping)
            )
        else:
            processed_dataset = dataset_zs
            
        logger.info(f"Number of examples in {dataset_name}: {len(processed_dataset)}")
        logger.info(f"Example from {dataset_name}: {processed_dataset[0]}")
        return processed_dataset
    
    def evaluate_model(self, nlp, dataset, dataset_name: str) -> Dict[str, Any]:
        """Evaluate the model on a specific dataset."""
        logger.info(f"Starting evaluation on {dataset_name}...")
        evaluation_results = evaluate(nlp, dataset, metric=Seqeval())
        logger.info(f"Evaluation results for {self.model_name} on {dataset_name}:")
        logger.info(evaluation_results)
        return evaluation_results
    
    def run_single_evaluation(self, dataset_name: str, mapping_file: str, output_dir: str) -> Dict[str, Any]:
        """Run evaluation on a single dataset."""
        logger.info(f"\n{'='*20} Evaluating {dataset_name.upper()} {'='*20}")
        
        # Load entity mapping for this dataset
        entity_mapping = self.load_entity_mapping(mapping_file, dataset_name)
        
        # Setup pipeline for this dataset
        nlp = self.setup_pipeline(entity_mapping)
        
        # Load and process dataset
        dataset = self.load_and_process_dataset(dataset_name, entity_mapping)
        
        # Evaluate model
        evaluation_results = self.evaluate_model(nlp, dataset, dataset_name)
        
        # Save individual results
        self.save_individual_results(evaluation_results, dataset_name, output_dir)
        
        return evaluation_results
    
    def run_evaluation(self, mapping_file: str, output_dir: str) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Run the complete evaluation pipeline."""
        datasets_to_evaluate = self.get_datasets_to_evaluate()
        
        if len(datasets_to_evaluate) == 1:
            # Single dataset evaluation
            results = self.run_single_evaluation(datasets_to_evaluate[0], mapping_file, output_dir)
            return results
        else:
            # Multiple datasets evaluation
            all_results = {}
            for dataset_name in datasets_to_evaluate:
                results = self.run_single_evaluation(dataset_name, mapping_file, output_dir)
                all_results[dataset_name] = results
            
            # Save combined summary for "all" datasets
            self.save_combined_summary(all_results, output_dir)
            return all_results
    
    def save_individual_results(self, evaluation_results: Dict[str, Any], dataset_name: str, output_dir: str) -> None:
        """Save evaluation results for an individual dataset."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        if self.is_gliner:
            detailed_file = f"{output_dir}/{self.model_name}_{dataset_name}_{self.split}_{self.type_case}.json"
        else:
            detailed_file = f"{output_dir}/{self.model_name}_{dataset_name}_{self.split}.json"
        
        save_to_json(evaluation_results, detailed_file)
        logger.info(f"Detailed results saved to {detailed_file}")
    
    def save_combined_summary(self, all_results: Dict[str, Dict[str, Any]], output_dir: str) -> None:
        """Save combined summary when evaluating all datasets."""
        # Create combined summary with dataset-specific keys
        combined_summary = {
            "model_name": self.model_name,
            "datasets": list(all_results.keys()),
            "split": self.split,
        }
        
        # Add type_case only for GLiNER models
        if self.is_gliner:
            combined_summary["type_case"] = self.type_case
        
        # Add metrics for each dataset
        for dataset_name, results in all_results.items():
            linker_results = results['linker']
            combined_summary[f"recall_macro_{dataset_name}"] = linker_results['overall_recall_macro']
            combined_summary[f"precision_macro_{dataset_name}"] = linker_results['overall_precision_macro']
            combined_summary[f"f1_macro_{dataset_name}"] = linker_results['overall_f1_macro']
        
        # Save combined summary
        if self.is_gliner:
            summary_file = f"{output_dir}/overall_all_{self.split}_{self.type_case}.jsonl"
        else:
            summary_file = f"{output_dir}/overall_all_{self.split}.jsonl"
        
        save_to_jsonl(combined_summary, summary_file)
        logger.info(f"Combined summary saved to {summary_file}")
    
    def save_single_summary(self, evaluation_results: Dict[str, Any], dataset_name: str, output_dir: str) -> None:
        """Save summary for single dataset evaluation (when dataset != 'all')."""
        # Only save individual summaries when not evaluating all datasets
        if self.dataset != "all":
            summary = {
                "model_name": self.model_name,
                "dataset": dataset_name,
                "split": self.split,
                "recall_macro": evaluation_results['linker']['overall_recall_macro'],
                "precision_macro": evaluation_results['linker']['overall_precision_macro'],
                "f1_macro": evaluation_results['linker']['overall_f1_macro'],
            }
            
            # Add type_case only for GLiNER models
            if self.is_gliner:
                summary["type_case"] = self.type_case
                summary_file = f"{output_dir}/overall_{dataset_name}_{self.split}_{self.type_case}.jsonl"
            else:
                summary_file = f"{output_dir}/overall_{dataset_name}_{self.split}.jsonl"
            
            save_to_jsonl(summary, summary_file)
            logger.info(f"Summary saved to {summary_file}")
    
    


def convert_and_round(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python types and round floats.
    
    Args:
        obj: Object to convert
        
    Returns:
        Converted object with rounded floats
    """
    if isinstance(obj, dict):
        return {k: convert_and_round(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_and_round(v) for v in obj]
    elif isinstance(obj, (np.float64, np.float32, float)):
        return round(float(obj), 2)
    elif isinstance(obj, (np.int64, np.int32, int)):
        return int(obj)
    else:
        return obj


def save_to_json(obj: Dict[str, Any], path: str) -> None:
    """Save object to JSON file with proper formatting."""
    obj_converted = convert_and_round(obj)
    with open(path, 'w') as f:
        json.dump(obj_converted, f, indent=4)


def save_to_jsonl(obj: Dict[str, Any], path: str) -> None:
    """Append object to JSONL file."""
    obj_converted = convert_and_round(obj)
    with open(path, 'a') as f:
        json.dump(obj_converted, f)
        f.write('\n')


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate NER models using zshot on multiple datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="urchade/gliner_large-v1",#"alecocc/zeroner-base", #"urchade/gliner_large-v1", #"alecocc/zeroner-base",
        help="HuggingFace model name"
    )
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="all",
        choices=["ontonotes", "medmentions", "legalner", "all"],
        help="Dataset name (use 'all' to evaluate on all datasets)"
    )
    
    parser.add_argument(
        "--split", "-s",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="Dataset split"
    )
    
    parser.add_argument(
        "--type-case", "-t",
        type=str,
        default="lowercase",
        choices=["lowercase", "title_case"],
        help="Entity type case for GLiNER models (ignored for SMXM models)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run the model on"
    )
    
    parser.add_argument(
        "--mapping-file",
        type=str,
        default="data/entity_type_mapping/gliner_types_map.json",#data/entity_type_mapping/zeroner_descr_map_v2.json", #"data/entity_type_mapping/gliner_types_map.json",
        help="Path to entity type mapping JSON file"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="out/evaluation_results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    # Check if mapping file exists
    if not os.path.exists(args.mapping_file):
        raise FileNotFoundError(f"Entity mapping file not found: {args.mapping_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.verbose:
        logger.info(f"Configuration:")
        logger.info(f"  Model: {args.model}")
        logger.info(f"  Dataset: {args.dataset}")
        logger.info(f"  Split: {args.split}")
        logger.info(f"  Type case: {args.type_case}")
        logger.info(f"  Device: {args.device}")
        logger.info(f"  Mapping file: {args.mapping_file}")
        logger.info(f"  Output directory: {args.output_dir}")


def print_summary(evaluator: NEREvaluator, results: Union[Dict[str, Any], Dict[str, Dict[str, Any]]], args: argparse.Namespace) -> None:
    """Print evaluation summary."""
    logger.info("\n" + "="*70)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*70)
    logger.info(f"Model: {evaluator.model_name}")
    logger.info(f"Split: {args.split}")
    
    if args.dataset == "all":
        logger.info(f"Datasets: {', '.join(evaluator.available_datasets)}")
        logger.info("-" * 70)
        for dataset_name, dataset_results in results.items():
            linker_results = dataset_results['linker']
            logger.info(f"{dataset_name.upper()}:")
            logger.info(f"  F1 Macro: {linker_results['overall_f1_macro']:.4f}")
            logger.info(f"  Precision Macro: {linker_results['overall_precision_macro']:.4f}")
            logger.info(f"  Recall Macro: {linker_results['overall_recall_macro']:.4f}")
    else:
        logger.info(f"Dataset: {args.dataset}")
        linker_results = results['linker']
        logger.info(f"F1 Macro: {linker_results['overall_f1_macro']:.4f}")
        logger.info(f"Precision Macro: {linker_results['overall_precision_macro']:.4f}")
        logger.info(f"Recall Macro: {linker_results['overall_recall_macro']:.4f}")
    
    logger.info("="*70)


def main() -> None:
    """Main function to run the NER evaluation."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validate arguments
        validate_arguments(args)

        # After parsing args, set up log filename
        now_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(f"logs/{now_dir}", exist_ok=True)
        log_filename = f"logs/{now_dir}/output_{args.model.split('/')[-1]}_{args.dataset}_{args.split}_{args.type_case}.log" if "gliner" in args.model.lower() else \
                        f"logs/{now_dir}/output_{args.model.split('/')[-1]}_{args.dataset}_{args.split}.log"

        file_handler = logging.FileHandler(log_filename, mode='w')
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        # Optional log
        logger.info(f"Log file created: {log_filename}")
        logger.info("Starting NER evaluation script")
        logger.info(f"Configuration: {args}")
        
        # Initialize evaluator
        evaluator = NEREvaluator(
            model_hf=args.model,
            dataset=args.dataset,
            split=args.split,
            type_case=args.type_case,
            device=args.device
        )
        
        # Run evaluation
        results = evaluator.run_evaluation(
            mapping_file=args.mapping_file,
            output_dir=args.output_dir
        )
        
        # Print summary
        print_summary(evaluator, results, args)
        
    except KeyboardInterrupt:
        logger.info("\nEvaluation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.info(f"Error during evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()