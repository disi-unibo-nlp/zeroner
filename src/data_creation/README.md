### How did we obtain the data?
The pretraining data was derived from the first 50,000 passages of the Pile uncensored dataset. The process involved:

- Splitting each passage into varying token lengths (using the BERT tokenizer), ranging from 30 to 300 tokens. This variation helps the student model adapt to different input lengths.
- Ignoring sentences from specific subsets, including: [Ubuntu IRC, DM Mathematics, EuroParl, GitHub, StackExchange]
- Removing all non-English text using [lingua-py](https://github.com/pemistahl/lingua-py).
  
To replicate our data preparation process, follow these steps:

Run the preprocessing script:
```bash
python3 src/create_data/pre_process.py
```
Execute the post-processing step:
```bash
python3 src/create_data/post_process.py
```
Generate annotations:
```bash
python3 src/create_data/gen_annotations.py
```
Convert the data to BIO format:
```bash
python3 src/create_data/convert_to_bio.py
```

### Self-Correction
In the second step of our data preparation, we employ a self-correction mechanism powered by the LLM itself. This mechanism evaluates data based on two key factors:

- Correctness: Only data with a correctness score of 3 is retained.
- Completeness: Data with a completeness score of at least 2 is kept.
