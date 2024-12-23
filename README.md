# ZeroNER 
This repository contains code and data for **ZeroNER**, a description-driven framework designed to enhance zero-shot Named Entity Recognition (NER) in low-resource settings. By leveraging **entity type descriptions** through cross-attention, ZeroNER enables a BERT-based student model to identify any entity type without requiring additional training.

Evaluated on three real-world zero-shot benchmarks under a rigorous *hard* zero-shot setting, ZeroNER consistently outperforms several Large Language Models (LLMs) by up to 15% in F1 score and surpasses alternative lightweight methods that rely solely on type names.

Our findings also reveal that many LLMs significantly benefit from the use of type descriptions, highlighting their potential in advancing zero-shot NER.

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


