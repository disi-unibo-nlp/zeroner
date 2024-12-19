We run LLMs evaluation on our benchmarks by using VLLM. 

To run inference and evaluation with Llama-3.1-8B-Instruct on all the benchmarks by leveraging entity type names:

```bash
python3 -m src.llm_inference.run_inference \
     --model_name "meta-llama/Llama-3.1-8B-Instruct" \
     --out_preds_dir "./output/llm/predictions" \
     --batch_size 100 \
     --datasets "OntoNotes-ZS,MedMentions-ZS,LegalNER-ZS"

python3 -m src.llm_inference.eval_preds \
     --model_name "meta-llama/Llama-3.1-8B-Instruct" \
     --datasets "OntoNotes-ZS,MedMentions-ZS,LegalNER-ZS" \
```

Instead, to run inference and evaluation with Llama-3.1-8B-Instruct on all the benchmarks by leveraging entity type descriptions:
```bash
python3 -m src.llm_inference.run_inference \
     --model_name "meta-llama/Llama-3.1-8B-Instruct" \
     --out_preds_dir "./output/llm/predictions" \
     --batch_size 100 \
     --w_descriptions

python3 -m src.llm_inference.eval_preds \
     --model_name "meta-llama/Llama-3.1-8B-Instruct" \
     --datasets "OntoNotes-ZS,MedMentions-ZS,LegalNER-ZS" \
     --w_descriptions
```

Example prompt with entity type names:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 17 Dec 2024

You are an expert entity classifier. You have to identify the entities within the given sentence that belong to one of the specified labels. 
There are no overlapped entities, so each word can belong to only one label. 
You can use only the possible labels provided. No other label is allowed.

Provide output in the following format: 
Return a list, marked with square brackets '[' and ']', containing string tuples. 
Each tuple should follow the pattern: ("entity", "label"). 
Prefix the entire list with '###entities:'. 
For example: 
###entities: [("entity 1", "label of entity 1"), ("entity 2", "label of entity 2"), ...]

If no entities are found, return an empty list like this: 
###entities: []

Don't add further information.<|eot_id|><|start_header_id|>user<|end_header_id|>

Possible labels: ['FAC', 'WORK_OF_ART', 'LOC']

Sentence: Before dawn on January 3 , a sewage pipe leakage accident occurred at the main and side roads of Jingguang Bridge , East Third Ring Road , Beijing Municipality , resulting in the road caving in .

Consider that FAC corresponds to Facility and LOC corresponds to Location.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

Example prompt with entity type descriptions:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 17 Dec 2024

You are an expert entity classifier. You have to identify the entities within the given sentence that belong to one of the specified labels. 
There are no overlapped entities, so each word can belong to only one label. 
You can use only the possible labels provided. No other label is allowed.

Provide output in the following format: 
Return a list, marked with square brackets '[' and ']', containing string tuples. 
Each tuple should follow the pattern: ("entity", "label"). 
Prefix the entire list with '###entities:'. 
For example: 
###entities: [("entity 1", "label of entity 1"), ("entity 2", "label of entity 2"), ...]

If no entities are found, return an empty list like this: 
###entities: []

Don't add further information.<|eot_id|><|start_header_id|>user<|end_header_id|>

Possible labels: 
LOC: Names of geographical locations other than GPEs. These include mountain ranges, coasts, borders, planets, geo-coordinates, bodies of water. Also included in this category are named regions such as the Middle East, areas, neighborhoods, continents and regions of continents. Do NOT mark deictics or other non-proper nouns: here, there, everywhere, etc. As with GPEs, directional modifiers such as "southern" are only marked when they are part of the location name itself.
FAC: Names of man-made structures: infrastructure (streets, bridges), buildings, monuments, etc. belong to this type. Buildings that are referred to using the name of the company or organization that uses them should be marked as FAC when they refer to the physical structure of the building itself, usually in a locative way: "I'm reporting live from right outside [Massachusetts General Hospital]
WORK_OF_ART: Titles of books, songs, television programs and other creations. Also includes awards. These are usually surrounded by quotation marks in the article (though the quotations are not included in the annotation). Newspaper headlines should only be marked if they are referential. In other words the headline of the article being annotated should not be marked but if in the body of the text here is a reference to an article, then it is markable as a work of art.

Sentence: Before dawn on January 3 , a sewage pipe leakage accident occurred at the main and side roads of Jingguang Bridge , East Third Ring Road , Beijing Municipality , resulting in the road caving in .<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```


