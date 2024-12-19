import json
import os
from collections import Counter
from tqdm import tqdm
from langdetect import detect
from lingua import Language, LanguageDetectorBuilder
input_path = "data/pretrain/tokenized/data_5.jsonl"

data_correct = []
with open(input_path) as f:
    for idx, line in enumerate(f):
        #print(f"Processing line {idx}: {line.strip()}")
        try:
            data = json.loads(line.strip())
            data_correct.append(data)
        except json.JSONDecodeError as e:
            print(f"Error on line {idx}: {e}")
            continue

print(len(data_correct))
print(data_correct[0]['subset'])
subsets = [el['subset'] for el in data_correct if 'subset' in el]
print(len(subsets))
print(Counter(subsets))
unique_subsets = set(subsets)
extracted_subsets = []

SUBSETS_TO_REMOVE = ['Ubuntu IRC', 'DM Mathematics', 'EuroParl', 'Github', 'StackExchange'] 
filtered_data = [el for el in data_correct if el['subset'] not in SUBSETS_TO_REMOVE]
filtered_data_2 = [el for el in filtered_data if el['actual_tokens'] <= el['max_tokens']]
filtered_data_3 = [el for el in filtered_data_2 if el['actual_tokens'] <= 300 or el['max_tokens'] <= 300]


languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH, Language.ITALIAN, Language.PORTUGUESE]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

count = 0
final_data = []
debug = False
for k, item in enumerate(tqdm(filtered_data_3)):
    if detector.detect_language_of(item['text']) == Language.ENGLISH:
        final_data.append(item)
    elif debug:
        print(item['subset'])
        print(item['text'])
        print(detector.detect_language_of(item['text']))
        print("---------------------------------------")

print("STARTING DATA:", len(data_correct))
print("FILTERED DATA after subsets cleansing:", len(filtered_data))
print("FILTERED DATA after token mismatch cleansing:", len(filtered_data_2))
print("FILTERED DATA after token mismatch cleansing:", len(filtered_data_3))
print("FILTERED DATA after language cleansing:", len(final_data))

os.makedirs('data/pretrain/processed', exist_ok=True)
for item in final_data:
    with open('data/pretrain/processed/data_processed_3.jsonl', 'a') as f:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")






            