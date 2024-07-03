from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import torch

model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

train_dataset = ... # Load the training dataset
val_dataset = ... # Load the validation dataset

def encode(batch):
    return tokenizer.prepare_seq2seq_batch(
        src_texts=batch['source_language_sentences'],
        tgt_texts=batch['target_language_sentences'],
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

train_dataset = train_dataset.map(encode, batched=True, batch_size=16)
val_dataset = val_dataset.map(encode, batched=True, batch_size=16)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='steps',
    eval_steps=100,
    save_total_limit=1,
    save_steps=500
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    tokenizer=tokenizer,
)

trainer.train()
