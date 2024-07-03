# https://github.com/huggingface/notebooks/blob/main/examples/translation.ipynb
import datasets
import pandas
import transformers
from datasets import Dataset
print(transformers.__version__)

# from transformers.utils import send_example_telemetry ##################
# send_example_telemetry("translation_notebook", framework="pytorch")

model_checkpoint = "./models/opus-mt-en-ro"  ### T5, mT5 ...
from datasets import load_dataset, load_metric

import pandas

# raw_datasets = {}
# data_files={"train": "./models/wmt16/ro-en/wmt16-train.arrow", 
#             "validation": "./models/wmt16/ro-en/wmt16-validation.arrow",
#             "test": "./models/wmt16/ro-en/wmt16-test.arrow"}
# raw_datasets['train'] = Dataset.from_file(data_files['train'])
# raw_datasets['validation'] = Dataset.from_file(data_files['validation'])
# raw_datasets['test'] = Dataset.from_file(data_files['test'])

# raw_datasets = pandas.DataFrame.from_dict(raw_datasets, orient='index').transpose()

# raw_datasets['train'] = Dataset.from_file("./wmt16/ro-en/wmt16-train.arrow")
raw_datasets = load_dataset("./models/wmt16")
# data_files={"train": "./wmt16/ro-en/wmt16-train.arrow"}
# raw_datasets = load_dataset("arrow", data_files=data_files)
# raw_datasets = load_dataset("arrow", data_files={"train": "wmt16/ro-en/wmt16-train.arrow", "test": "wmt16/ro-en/wmt16-test.arrow"})

print(raw_datasets)
print(raw_datasets["train"][0])
# print(raw_datasets)

# metric = load_metric("sacrebleu")
# print(metric)

# import datasets
# import random
# import pandas as pd
# from IPython.display import display, HTML

# def show_random_elements(dataset, num_examples=5):
#     assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
#     picks = []
#     for _ in range(num_examples):
#         pick = random.randint(0, len(dataset) - 1)
#         while pick in picks:
#             pick = random.randint(0, len(dataset) - 1)
#         picks.append(pick)

#     df = pd.DataFrame(dataset[picks])
#     for column, typ in dataset.features.items():
#         if isinstance(typ, datasets.ClassLabel):
#             df[column] = df[column].transform(lambda i: typ.names[i])
#     display(HTML(df.to_html()))

# show_random_elements(raw_datasets["train"])
# print(metric)

# fake_preds = ["hello there", "general kenobi"]
# fake_labels = [["hello there"], ["general kenobi"]]
# print(metric.compute(predictions=fake_preds, references=fake_labels))

# # Preprocessing the data
#
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
#
# if "mbart" in model_checkpoint:
#     tokenizer.src_lang = "en-XX"
#     tokenizer.tgt_lang = "ro-RO"
#
# print(tokenizer("Hello, this one sentence!"))
# print(tokenizer(["Hello, this one sentence!", "This is another sentence."]))
#
# with tokenizer.as_target_tokenizer():
#     print(tokenizer(["Hello, this one sentence!", "This is another sentence."]))
#
# if model_checkpoint in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
#     prefix = "translate English to Romanian: "
# else:
#     prefix = ""
#
#
# max_input_length = 128
# max_target_length = 128
# source_lang = "en"
# target_lang = "ro"
#
# def preprocess_function(examples):
#     inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
#     targets = [ex[target_lang] for ex in examples["translation"]]
#     model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
#
#     # Setup the tokenizer for targets
#     with tokenizer.as_target_tokenizer():
#         labels = tokenizer(targets, max_length=max_target_length, truncation=True)
#
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs
#
# print(preprocess_function(raw_datasets['train'][:2]))
#
# tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
# print(tokenized_datasets)

# # Fine-tuning the model
# from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
#
# model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
# batch_size = 16
# model_name = model_checkpoint.split("/")[-1]
# args = Seq2SeqTrainingArguments(
#     f"{model_name}-finetuned-{source_lang}-to-{target_lang}",
#     evaluation_strategy = "epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     weight_decay=0.01,
#     save_total_limit=3,
#     num_train_epochs=1,
#     predict_with_generate=True,
#     fp16=True,
#     push_to_hub=False,
# )
# data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
# import numpy as np
#
# def postprocess_text(preds, labels):
#     preds = [pred.strip() for pred in preds]
#     labels = [[label.strip()] for label in labels]
#
#     return preds, labels
#
# def compute_metrics(eval_preds):
#     preds, labels = eval_preds
#     if isinstance(preds, tuple):
#         preds = preds[0]
#     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
#
#     # Replace -100 in the labels as we can't decode them.
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#
#     # Some simple post-processing
#     decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
#
#     result = metric.compute(predictions=decoded_preds, references=decoded_labels)
#     result = {"bleu": result["score"]}
#
#     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
#     result["gen_len"] = np.mean(prediction_lens)
#     result = {k: round(v, 4) for k, v in result.items()}
#     return result
#
# trainer = Seq2SeqTrainer(
#     model,
#     args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["validation"],
#     data_collator=data_collator,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )
#
# trainer.train()
#
# trainer.save_model(output_dir='sgugger/my-awesome-model')