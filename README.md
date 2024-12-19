# ZeroNER 
This repository contains code and data for **ZeroNER**, a description-driven framework designed to enhance zero-shot Named Entity Recognition (NER) in low-resource settings. By leveraging **entity type descriptions** through cross-attention, ZeroNER enables a BERT-based student model to identify any entity type without requiring additional training.

Evaluated on three real-world zero-shot benchmarks under a rigorous *hard* zero-shot setting, ZeroNER consistently outperforms several Large Language Models (LLMs) by up to 15% in F1 score and surpasses alternative lightweight methods that rely solely on type names.

Our findings also reveal that many LLMs significantly benefit from the use of type descriptions, highlighting their potential in advancing zero-shot NER.

## Pretrain Data
Pretraining data in BIO format is available for download [here](https://drive.google.com/file/d/1slUHvSIP0yrzNJBIJivBRWe0Z10fjlM1/view?usp=sharing).

You can load the data into a dataset using the following code:
```python
# pip install datasets
from datasets import Dataset
ds = Dataset.from_json('pretrain_data.jsonl')
```
