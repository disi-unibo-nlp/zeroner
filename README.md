# ZeroNER 
This repository contains code and data for **ZeroNER**, a description-driven framework designed to enhance zero-shot Named Entity Recognition (NER) in low-resource settings. By leveraging **entity type descriptions** through cross-attention, ZeroNER enables a BERT-based student model to identify any entity type without requiring additional training.

Evaluated on three real-world zero-shot benchmarks under a rigorous *hard* zero-shot setting, ZeroNER consistently outperforms several Large Language Models (LLMs) by up to 15% in F1 score and surpasses alternative lightweight methods that rely solely on type names.

Our findings also reveal that many LLMs significantly benefit from the use of type descriptions, highlighting their potential in advancing zero-shot NER.

---
## Quick Start

The **ZeroNER checkpoint** is available for download [here](https://drive.google.com/drive/folders/1VAPfd5xzir-4vxj-P5l4j9_4wbdchuRi?usp=sharing).

We have integrated our model within the [IBM Zshot library](https://github.com/IBM/zshot) for a quick and easy use for the user. Also, a full support is allowed for OntoNotes-ZS dataset with corresponding descriptions. Process is still ongoing and we plan to integrate both MedMentions-ZS and LegalNER-ZS, as well as future datasets.

```python
import spacy
import datasets

from zshot import PipelineConfig, displacy
from zshot.linker import LinkerSMXM
from zshot.utils.data_models import Entity


entities = [
    Entity(name='FAC', description='Names of man-made structures: infrastructure (streets, bridges), buildings, monuments, etc. belong to this type. Buildings that are referred to using the name of the company or organization that uses them should be marked as FAC when they refer to the physical structure of the building itself, usually in a locative way: "I\'m reporting live from right outside [Massachusetts General Hospital]"', vocabulary=None),
    Entity(name='LOC', description='Names of geographical locations other than GPEs. These include mountain ranges, coasts, borders, planets, geo-coordinates, bodies of water. Also included in this category are named regions such as the Middle East, areas, neighborhoods, continents and regions of continents. Do NOT mark deictics or other non-proper nouns: here, there, everywhere, etc. As with GPEs, directional modifiers such as "southern" are only marked when they are part of the location name itself.', vocabulary=None),
    Entity(name='WORK_OF_ART', description='Titles of books, songs, television programs and other creations. Also includes awards. These are usually surrounded by quotation marks in the article (though the quotations are not included in the annotation). Newspaper headlines should only be marked if they are referential. In other words the headline of the article being annotated should not be marked but if in the body of the text here is a reference to an article, then it is markable as a work of art.', vocabulary=None)
]

nlp = spacy.blank("en")
nlp_config = PipelineConfig(
    linker=LinkerSMXM(model_name="zeroner_base"),
    entities=entities,
    device='cuda'
)

nlp.add_pipe("zshot", config=nlp_config, last=True)

text = """
I remember the SMS was written like this at that time , saying that , ah , there was a sewage pipe leakage accident on the side road at the southeast corner of Jingguang Bridge at East Third Ring Road , and , well , traffic supervision was implemented near Chaoyang Road , Jingguang Bridge , and East Third Ring Road , and requesting cars to make a detour .
"""

doc = nlp(text)
displacy.serve(doc, style="ent")
```
Output:

---

## Repository Overview
This repository is organized as follows:

- **[`src/data_creation`](src/data_creation)**: Contains scripts and details for the creation of pretraining data, including preprocessing, annotation generation, and BIO conversion.
- **[`src/llm_inference`](src/llm_inference)**: Includes code for running LLM inference and evaluation over datasets used in our experiments.
- **[`src/zeroner`](src/zeroner)**: Core implementation of the ZeroNER framework (available soon).

---

## Pretrain Data
Pretraining data in BIO format is available for download [here](https://drive.google.com/file/d/1slUHvSIP0yrzNJBIJivBRWe0Z10fjlM1/view?usp=sharing).

You can load the data into a dataset using the following code:
```python
# pip install datasets
from datasets import Dataset
ds = Dataset.from_json('pretrain_data.jsonl')
```

### How did we obtain the data?
The pretraining data was derived from the first 50,000 passages of the Pile uncensored dataset. The process involved:

- Splitting each passage into varying token lengths (using the BERT tokenizer), ranging from 30 to 300 tokens. This variation helps the student model adapt to different input lengths.
- Ignoring sentences from specific subsets, including: [Ubuntu IRC, DM Mathematics, EuroParl, GitHub, StackExchange]
- Removing all non-English text using [lingua-py](https://github.com/pemistahl/lingua-py).

---

## Model Checkpoint 
The ZeroNER checkpoint is available for download [here](https://drive.google.com/drive/folders/1VAPfd5xzir-4vxj-P5l4j9_4wbdchuRi?usp=sharing).


