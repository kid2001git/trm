import transformers
from datasets import load_dataset
print(transformers.__version__)

from datasets import load_dataset, load_metric

raw_datasets = load_dataset("wmt16", "ro-en")
metric = load_metric("sacrebleu")

print(raw_datasets)
# print(metric)

print(raw_datasets['train'][0])

import datasets
import random
import pandas as pd
# from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=5):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    # display(HTML(df.to_html()))

show_random_elements(raw_datasets["train"])
