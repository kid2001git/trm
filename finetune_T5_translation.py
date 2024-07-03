# https://huggingface.co/docs/transformers/tasks/translation


from datasets import load_dataset
from huggingface_hub import notebook_login

from huggingface_hub.hf_api import HfFolder


books = load_dataset("opus_books", "en-fr")

books = books["train"].train_test_split(test_size=0.2)

print(books["train"][0])

from transformers import AutoTokenizer

# checkpoint = "google-t5/t5-small"
checkpoint = "./t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

source_lang = "en"
target_lang = "fr"
prefix = "translate English to French: "


def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

tokenized_books = books.map(preprocess_function, batched=True)

HfFolder.save_token('tokenized_books')

print(tokenized_books['train'][0])

from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

import evaluate

metric = evaluate.load("sacrebleu")

import numpy as np


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
training_args = Seq2SeqTrainingArguments(
    output_dir="my_awesome_opus_books_model", 
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_books["train"],
    eval_dataset=tokenized_books["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# # to local
# trainer.push_to_hub()
trainer.save_model('T5_finetuned')

# # Inference
# text = "translate English to French: Legumes share resources with nitrogen-fixing bacteria."
# from transformers import pipeline

# translator = pipeline("translation_xx_to_yy", model="my_awesome_opus_books_model")
# translator(text)

# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("my_awesome_opus_books_model")
# inputs = tokenizer(text, return_tensors="pt").input_ids

# from transformers import AutoModelForSeq2SeqLM

# model = AutoModelForSeq2SeqLM.from_pretrained("my_awesome_opus_books_model")
# outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
# tokenizer.decode(outputs[0], skip_special_tokens=True)