from datasets import load_dataset
from collections import Counter
from tqdm import tqdm
import difflib
import re

id2tag = {'T058': "Health_Care_Activity", "T062": "Research_Activity", "T037": "Injury_or_Poisoning",
            "T038": "Biologic_Function", "T005": "Virus", "T007": "Bacterium", "T204": "Eukaryote",
            "T017": "Anotomical_Structure", "T074": "Medical_Device", "T031": "Body_Substance", "T103": "Chemical",
            "T168": "Food", "T201": "Clinical_Attribute", "T033": "Finding", "T082": "Spatial_Concept",
            "T022": "Body_System", "T091": "Biomedical_Occupation_or_Discipline", "T092": "Organization",
            "T097": "Professional_or_Occupational_Group", "T098": "Population_Group", "T170": "Intellectual_Product",
            "NEG": "NEG"}
def convert_to_tag(example):
  tokens = example['tokens']
  ner_tags = example['ner_tags']
  example = [ (x[:2] + id2tag[x[2:]]).replace("_", " ") if x != 'O' else 'O' for x in ner_tags]
  return {'tokens': tokens, 'ner_tags': example}

def normalize_tag(tag):
    """Normalize entity tags for better comparison"""
    if tag == 'O':
        return tag
    # Remove BIO prefixes and normalize
    normalized = tag[2:].lower().replace("_", " ").replace("-", " ").strip()
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized

def calculate_similarity(str1, str2):
    """Calculate similarity between two strings using difflib"""
    return difflib.SequenceMatcher(None, str1, str2).ratio()

def is_word_boundary_match(tag1, tag2):
    """Check if one tag is contained in another at word boundaries"""
    words1 = set(tag1.split())
    words2 = set(tag2.split())
    return bool(words1.intersection(words2))

def analyze_contamination(all_tags, benchmark_tags, similarity_threshold=0.9, min_word_length=2):
    """
    Comprehensive contamination analysis
    """
    results = {
        'exact_matches': {},
        'fuzzy_matches': {},
        'derivative_matches': {},
        'word_boundary_matches': {}
    }
    
    # Convert to sets for faster lookup
    all_tags_set = set(all_tags)
    benchmark_tags_set = set(benchmark_tags)
    
    # Create counter for training tags
    counter = Counter(all_tags)
    
    print("🔍 CONTAMINATION ANALYSIS")
    print("=" * 60)
    
    # 1. Exact matches
    exact_matches = benchmark_tags_set.intersection(all_tags_set)
    if exact_matches:
        print(f"❌ EXACT MATCHES FOUND: {len(exact_matches)}")
        for match in sorted(exact_matches):
            if match != 'O':  # Skip 'O' tags
                count = counter.get(match, 0)
                results['exact_matches'][match] = count
                print(f"  • {match}: {count} instances")
    else:
        print("✅ No exact matches found")
    
    print()
    
    # 2. Fuzzy matches (high similarity but not exact)
    print(f"🔍 FUZZY MATCHES (similarity >= {similarity_threshold}):")
    fuzzy_found = False
    already_categorized = set(exact_matches)
    
    for bench_tag in tqdm(benchmark_tags_set, desc="Checking fuzzy matches"):
        if bench_tag == 'O' or bench_tag in already_categorized:
            continue
            
        for train_tag in all_tags_set:
            if train_tag == 'O' or len(train_tag.strip()) < min_word_length:
                continue
                
            similarity = calculate_similarity(bench_tag, train_tag)
            if similarity >= similarity_threshold and bench_tag != train_tag:
                if bench_tag not in results['fuzzy_matches']:
                    results['fuzzy_matches'][bench_tag] = []
                results['fuzzy_matches'][bench_tag].append((train_tag, counter[train_tag], similarity))
                fuzzy_found = True

    # Update categorized set
    already_categorized.update(results['fuzzy_matches'].keys())
    
    if fuzzy_found:
        for bench_tag, matches in results['fuzzy_matches'].items():
            total_instances = sum(match[1] for match in matches)
            print(f"  • {bench_tag} → {total_instances} total instances")
            for train_tag, count, sim in sorted(matches, key=lambda x: x[2], reverse=True)[:5]:
                print(f"    - {train_tag}: {count} instances (sim: {sim:.2f})")
    else:
        print("  ✅ No fuzzy matches found")
    
    print()
    
    # 3. Word boundary matches
    print("🔍 WORD BOUNDARY MATCHES:")
    boundary_found = False
    
    for bench_tag in tqdm(benchmark_tags_set, desc="Checking word boundaries"):
        if bench_tag == 'O' or bench_tag in already_categorized:
            continue
            
        for train_tag in all_tags_set:
            if train_tag == 'O' or len(train_tag.strip()) < min_word_length:
                continue
            if train_tag in [match[0] for matches in results['fuzzy_matches'].get(bench_tag, []) for match in [matches]]:
                continue  # Skip if already found in fuzzy matches
                
            if is_word_boundary_match(bench_tag, train_tag) and bench_tag != train_tag:
                if bench_tag not in results['word_boundary_matches']:
                    results['word_boundary_matches'][bench_tag] = []
                results['word_boundary_matches'][bench_tag].append((train_tag, counter[train_tag]))
                boundary_found = True

    # Update categorized set
    already_categorized.update(results['word_boundary_matches'].keys())
    
    if boundary_found:
        for bench_tag, matches in results['word_boundary_matches'].items():
            total_instances = sum(match[1] for match in matches)
            print(f"  • {bench_tag} → {total_instances} total instances")
            for train_tag, count in sorted(matches, key=lambda x: x[1], reverse=True)[:10]:
                print(f"    - {train_tag}: {count} instances")
    else:
        print("  ✅ No word boundary matches found")
    
    print()
    
    # 4. Substring matches (your original derivative logic, but refined)
    print("🔍 SUBSTRING MATCHES:")
    substring_found = False
    
    for bench_tag in tqdm(benchmark_tags_set, desc="Checking substrings"):
        if bench_tag == 'O' or bench_tag in already_categorized:
            continue
            
        for train_tag in all_tags_set:
            if train_tag == 'O' or len(train_tag.strip()) < min_word_length:
                continue
                
            # Check substring relationship
            if (bench_tag in train_tag or train_tag in bench_tag) and bench_tag != train_tag:
                if bench_tag not in results['derivative_matches']:
                    results['derivative_matches'][bench_tag] = []
                results['derivative_matches'][bench_tag].append((train_tag, counter[train_tag]))
                substring_found = True
    
    if substring_found:
        for bench_tag, matches in results['derivative_matches'].items():
            total_instances = sum(match[1] for match in matches)
            print(f"  • {bench_tag} → {total_instances} total instances")
            for train_tag, count in sorted(matches, key=lambda x: x[1], reverse=True)[:10]:
                print(f"    - {train_tag}: {count} instances")
    else:
        print("  ✅ No substring matches found")
    
    return results

def calculate_contamination_metrics(results, total_benchmark_tags):
    """Calculate various contamination metrics"""
    print("\n📊 CONTAMINATION METRICS")
    print("=" * 60)
    
    total_contaminated = len(set().union(
        results['exact_matches'].keys(),
        results['fuzzy_matches'].keys(),
        results['word_boundary_matches'].keys(),
        results['derivative_matches'].keys()
    ))
    
    contamination_rate = total_contaminated / total_benchmark_tags if total_benchmark_tags > 0 else 0
    
    print(f"Total benchmark entity types: {total_benchmark_tags}")
    print(f"Contaminated entity types: {total_contaminated}")
    print(f"Contamination rate: {contamination_rate:.2%}")
    
    # Calculate severity scores
    severity_scores = {
        'exact': len(results['exact_matches']) * 1.0,
        'fuzzy': len(results['fuzzy_matches']) * 0.8,
        'word_boundary': len(results['word_boundary_matches']) * 0.6,
        'substring': len(results['derivative_matches']) * 0.4
    }
    
    total_severity = sum(severity_scores.values())
    max_possible_severity = total_benchmark_tags * 1.0
    
    severity_score = total_severity / max_possible_severity if max_possible_severity > 0 else 0
    
    print(f"Contamination severity score: {severity_score:.2%}")
    print(f"  - Exact matches (weight 1.0): {severity_scores['exact']}")
    print(f"  - Fuzzy matches (weight 0.8): {severity_scores['fuzzy']}")
    print(f"  - Word boundary matches (weight 0.6): {severity_scores['word_boundary']}")
    print(f"  - Substring matches (weight 0.4): {severity_scores['substring']}")
    
    return contamination_rate, severity_score

# Main execution
# Note: You need to define 'all_tags' - your training data entity tags
# all_tags = [...] # Your training entity tags go here
if __name__ == "__main__":
    DATASET = "pile_mistral"  # Change this to your dataset name if needed
    BENCHMARK = "legalner"  # Change this to your benchmark dataset name if needed

    if "pile_ner" in DATASET:
        print("Loading 'Pile-NER-type-IOB' dataset...")
        data = load_dataset("alecocc/Pile-NER-type-IOB", split="train")

    elif "pile_mistral" in DATASET:
        print("Loading 'pile-mistral-v0.1-IOB' dataset...")
        data = load_dataset("alecocc/pile-mistral-v0.1-IOB", split="train")
    
    else:
        raise ValueError(f"Unknown dataset: {DATASET}")
        
    

    all_tags = []
    for item in data:
        all_tags.extend([el.lower()[2:] for el in item['ner_tags'] if el.startswith("B-")])


    for split in ["validation", "test"]:
        print(f"\n🎯 ANALYZING SPLIT: {split.upper()}")
        print("=" * 80)
        
        # Load benchmark dataset
        #bench = load_dataset("alecocc/LegalNER-ZS", split=split)
        if "ontonotes" in BENCHMARK:
            from zshot.evaluation.dataset import load_ontonotes_zs
            print("Loading Ontonotes Zero-Shot benchmark dataset...")
            bench = load_ontonotes_zs(split=split)
        elif "medmentions" in BENCHMARK:
            from zshot.evaluation.dataset import load_medmentions_zs
            print("Loading MedMentions Zero-Shot benchmark dataset...")
            bench = load_medmentions_zs(split=split)
            bench = bench.map(convert_to_tag)
        elif "legalner" in BENCHMARK:
            print("Loading LegalNER Zero-Shot benchmark dataset...")
            bench = load_dataset("alecocc/LegalNER-ZS", split=split)
        else:
            raise ValueError(f"Unknown benchmark dataset: {BENCHMARK}")
        
        # Extract and normalize benchmark tags
        all_tags_bench = []
        for item in bench:
            normalized_tags = [normalize_tag(tag) for tag in item['ner_tags']]
            all_tags_bench.extend(normalized_tags)
        
        # Remove 'O' tags for analysis
        benchmark_entity_tags = [tag.replace("loc", "location").replace("fac", "facility").replace("norp", "national religious political").replace("org", "organization") for tag in set(all_tags_bench) if tag != 'O']
        
        print(f"Benchmark entity types found: {len(benchmark_entity_tags)}")
        print(f"Unique benchmark tags: {sorted(benchmark_entity_tags)}")
        print()
        
        # Perform contamination analysis
        # Note: Replace 'all_tags' with your actual training data tags
        results = analyze_contamination(all_tags, benchmark_entity_tags)
        contamination_rate, severity_score = calculate_contamination_metrics(
            results, len(benchmark_entity_tags)
        )
        
        #print("⚠️  IMPORTANT: Replace 'all_tags' with your training data entity tags to run the analysis")
        print("=" * 80)