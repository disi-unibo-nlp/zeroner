# ZeroNER 

## Pretrain Data
Pretraining data in BIO format is available for download [here](https://drive.google.com/file/d/1slUHvSIP0yrzNJBIJivBRWe0Z10fjlM1/view?usp=sharing).

You can load the data into a dataset using the following code:
```python
# pip install datasets
from datasets import Dataset
ds = Dataset.from_json('pretrain_data.jsonl')
```
