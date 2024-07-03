from datasets import load_dataset

emotion_dataset  = load_dataset('emotion')
print(emotion_dataset)
print(emotion_dataset['train'][0])

emotion_df = emotion_dataset['train'].to_pandas()
print(emotion_df.head())

features = emotion_dataset['train'].features
print(features)
print(features['label'].int2str(2))

id2label = {idx:features['label'].int2str(idx) for idx in range(6)}
print(id2label)

label2id = {v:k for k,v in id2label.items()}
print(label2id)

print(emotion_df['label'].value_counts(normalize=True).sort_index())

# Tokenize 
from transformers import AutoTokenizer

# model_ckpt = 'microsoft/MiniLM-L12-H384-uncased'
model_ckpt = 'MiniLM-L12-H384-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
print(emotion_dataset['train']['text'][:1])
print(tokenizer(emotion_dataset['train']['text'][:1]))

def tokenize_text(examples):
    return tokenizer(examples['text'], truncation=True, max_length=512)

emotion_dataset = emotion_dataset.map(tokenize_text, batched=True)
print(emotion_dataset)

class_weights = (1-(emotion_df['label'].value_counts().sort_index() / len(emotion_df))).values
print(class_weights)

import torch
class_weights = torch.from_numpy(class_weights).float().to('cuda')
print(class_weights)

from torch import nn
from transformers import Trainer

class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Feed inputs to model and extract logits
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # extract labels
        labels = inputs.get('labels')
        #loss func
        loss_func = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_func(logits, labels)
        return (loss, outputs) if return_outputs else loss

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=6, \
    id2label=id2label, label2id=label2id)

from sklearn.metrics import f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.prediction.argmax(-1)
    f1 = f1_score(labels, preds, average='weighted')
    return {'f1': f1}


from transformers import TrainingArguments

batch_size = 64
logging_steps = len(emotion_dataset['train'])
output_dir = 'minilm-finetuned-emotion'
training_args = TrainingArguments(output_dir=output_dir, num_train_epochs=5, learning_rate=2e-5,
    per_device_train_batch_size = batch_size, per_device_eval_batch_size=batch_size, 
    weight_decay = 0.01, evaluation_strategy='epoch', logging_steps=logging_steps,
    fp16=True, push_to_hub=False)

trainer = WeightedLossTrainer(model=model, 
                            args=training_args,
                            compute_metrics=compute_metrics,
                            train_dataset=emotion_dataset['train'],
                            tokenizer=tokenizer)
trainer.train()
