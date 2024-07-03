from datasets import load_dataset
import datasets
raw_datasets = load_dataset("kde4", lang1="en", lang2="fr")
# raw_datasets = load_dataset("./models/datasets/kde4")
print(datasets.__version__)

print(raw_datasets)

split_datasets = raw_datasets["train"].train_test_split(train_size=0.9, seed=20)
print(split_datasets)

split_datasets["validation"] = split_datasets.pop("test")

print(split_datasets["train"][1]["translation"])

from transformers import pipeline

# model_checkpoint = "./models/opus-mt-en-fr"
model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
translator = pipeline("translation", model=model_checkpoint)
translator("Default to expanded threads")

print(split_datasets["train"][172]["translation"])

print(translator("Unable to import %1 using the OFX importer plugin. This file is not the correct format."))

# data processing
from transformers import AutoTokenizer

# model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")

en_sentence = split_datasets["train"][1]["translation"]["en"]
fr_sentence = split_datasets["train"][1]["translation"]["fr"]

inputs = tokenizer(en_sentence, text_target=fr_sentence)
print(inputs)

wrong_targets = tokenizer(fr_sentence)
print(tokenizer.convert_ids_to_tokens(wrong_targets["input_ids"]))
print(tokenizer.convert_ids_to_tokens(inputs["labels"]))

max_length = 128


def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["fr"] for ex in examples["translation"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    return model_inputs

tokenized_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names,
)

from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
batch = data_collator([tokenized_datasets["train"][i] for i in range(1, 3)])
print(batch.keys())

for i in range(1, 3):
    print(tokenized_datasets["train"][i]["labels"])

# tf_train_dataset = model.prepare_tf_dataset(
#     tokenized_datasets["train"],
#     collate_fn=data_collator,
#     shuffle=True,
#     batch_size=32,
# )
# tf_eval_dataset = model.prepare_tf_dataset(
#     tokenized_datasets["validation"],
#     collate_fn=data_collator,
#     shuffle=False,
#     batch_size=16,
# )

# metric
import evaluate
metric = evaluate.load("sacrebleu")

predictions = [
    "This plugin lets you translate web pages between several languages automatically."
]
references = [
    [
        "This plugin allows you to automatically translate web pages between several languages."
    ]
]
results = metric.compute(predictions=predictions, references=references)

print(results)

predictions = ["This This This This"]
references = [
    [
        "This plugin allows you to automatically translate web pages between several languages."
    ]
]
metric.compute(predictions=predictions, references=references)




