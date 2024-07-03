import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
from transformers.pipelines.pt_utils import KeyDataset
import tqdm
import datasets
from datasets import Dataset
import os
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

os.environ["TOKENIZERS_PARALLELISM"] = "true"

BLEU = 'bleu'

language_mapping = {"es":"Spanish", "de":"German", "fr": "French", "it":"Italian", "pt":"Portuguese"}


print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

seed = 100
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


model_id = "./models/flan-t5-base" # Hugging Face Model Id
tokenizer = T5Tokenizer.from_pretrained(model_id)
model = T5ForConditionalGeneration.from_pretrained(model_id, device_map="auto")


# ## Load dataset and process the data

# The dataset has the following columns: 
# - `ID`
# - `input_to_translate`: the source sentence in English
# - `label`: the translation reference in the target language
# - `gender`: f(emale) or m(ale)
# - `language_pair`: `<source>_<target>`, such as en_fr for English to French

training_features = pd.read_csv("data/training.csv", encoding="utf-8-sig")
training_features.head(2)

def generate_prompt(x):
    language_mapping = {"es":"Spanish", "de":"German", "fr": "French", "it":"Italian", "pt":"Portuguese"}
    source_text = x["input_to_translate"]
    language = x["language_pair"].split('_')[1]
    input_text = f"Translate the following sentence from English to {language_mapping[language]}: \"{source_text}\" "
    return input_text


training_features["prompt"] = training_features.apply(generate_prompt, axis=1)
print(training_features.head(2))


# # #### Check the generated prompt:

# # In[8]:


# training_features.iloc[0]["prompt"]


# # In[10]:


# training_features.iloc[1]["prompt"]


# # #### Load and generate prompt for test set
# # 
# # The test set is smilar with the training set, except that it is lacking the "label" column.

# # In[13]:


# test_features = pd.read_csv("data/test_features.csv", encoding="utf-8-sig")
# list(test_features)


# # In[12]:


# test_features["prompt"] = test_features.apply(generate_prompt, axis=1)
# list(test_features)


# # ### Use Hugging Face Dataset object

# # In[11]:


# train_ds_raw = datasets.Dataset.from_pandas(training_features, split="train")
# train_ds_raw


# # In[12]:


# test_ds_raw = datasets.Dataset.from_pandas(test_features, split="test")
# test_ds_raw


# # In[17]:


# tokenizer.model_max_length


# # In[18]:


# tokenizer.pad_token_id


# # ### Figure out token length and tokenize the training set

# # In[19]:


# tokenized_source_training = train_ds_raw.map(
#     lambda x: tokenizer(x["prompt"], truncation=True), 
#     batched=True, remove_columns=['ID', 'input_to_translate', 'label', 'gender', 'language_pair', 'prompt'])

# source_lengths_training = [len(x) for x in tokenized_source_training["input_ids"]]

# print(f"Max source length: {max(source_lengths_training)}")
# print(f"95% source length: {int(np.percentile(source_lengths_training, 95))}")


# # In[20]:


# tokenized_target_training = train_ds_raw.map(
#     lambda x: tokenizer(x["label"], truncation=True), 
#     batched=True, remove_columns=['ID', 'input_to_translate', 'label', 'gender', 'language_pair', 'prompt'])
# target_lengths_training = [len(x) for x in tokenized_target_training["input_ids"]]

# print(f"Max target length: {max(target_lengths_training)}")
# print(f"95% target length: {int(np.percentile(target_lengths_training, 95))}")


# # In[21]:


# tokenized_source_test = test_ds_raw.map(
#     lambda x: tokenizer(x["prompt"], truncation=True), 
#     batched=True, remove_columns=['ID', 'input_to_translate', 'gender', 'language_pair', 'prompt'])

# source_lengths_test = [len(x) for x in tokenized_source_test["input_ids"]]

# print(f"Max source length in test set: {max(source_lengths_test)}")
# print(f"95% source length in test set: {int(np.percentile(source_lengths_test, 95))}")


# # In[22]:


# max_source_length = max(max(source_lengths_training), max(source_lengths_test))
# max_source_length


# # In[23]:


# max_target_length = max(target_lengths_training)
# max_target_length


# # In[24]:


# # reference: https://www.philschmid.de/fine-tune-flan-t5-deepspeed
# def preprocess_function(sample, padding="max_length"):

#     # tokenize inputs
#     model_inputs = tokenizer(sample["prompt"], max_length=max_source_length, padding=padding, truncation=True)

#     # Tokenize targets with the `text_target` keyword argument
#     labels = tokenizer(text_target=sample["label"], max_length=max_target_length, padding=padding, truncation=True)

#     # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
#     # padding in the loss.
#     if padding == "max_length":
#         labels["input_ids"] = [
#             [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
#         ]

#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs


# # In[25]:


# tokenized_train_ds = train_ds_raw.map(
#     preprocess_function, batched=True, 
#     remove_columns=['ID', 'input_to_translate', 'label', 'gender', 'language_pair', 'prompt'])


# # In[26]:


# tokenized_train_ds


# # ### Split the original training set into train and test:

# # In[27]:


# ds_dict = tokenized_train_ds.train_test_split(test_size=0.2)
# ds_dict


# # In[28]:


# trainset = ds_dict["train"]
# trainset           


# # In[29]:


# testset = ds_dict["test"]
# testset


# # In[30]:


# # save dataset to disk
# save_dataset_path = "training_data"
# trainset.save_to_disk(os.path.join(save_dataset_path,"train"))
# testset.save_to_disk(os.path.join(save_dataset_path,"eval"))


# # ## Fine-tuning the model
# # 
# # reference: https://www.philschmid.de/fine-tune-flan-t5-deepspeed 

# # In[31]:


# # !pip3 install -q pytesseract transformers datasets nltk tensorboard py7zr evaluate sacrebleu --upgrade


# # In[32]:


# import evaluate
# import nltk
# import numpy as np
# from nltk.tokenize import sent_tokenize
# nltk.download("punkt")


# # In[33]:


# # Metric
# metric = evaluate.load("sacrebleu")


# # In[34]:


# # helper function to postprocess text
# def postprocess_text(preds, labels):
#     preds = [pred.strip() for pred in preds]
#     labels = [[label.strip()] for label in labels]
#     return preds, labels

# def compute_metrics(eval_preds):
#     preds, labels = eval_preds
#     if isinstance(preds, tuple):
#         preds = preds[0]

#     # Replace -100 in the labels as we can't decode them.
#     # for some reason, also get a lot of -100 in preds
#     preds = np.where(preds != -100, preds, tokenizer.pad_token_id)    
#     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

#     # Replace -100 in the labels as we can't decode them.
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#     # Some simple post-processing
#     decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

#     result = metric.compute(predictions=decoded_preds, references=decoded_labels)
#     result = {"bleu": result["score"]}

#     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
#     result["gen_len"] = np.mean(prediction_lens)
#     result = {k: round(v, 4) for k, v in result.items()}
#     return result


# # In[35]:


# from transformers import DataCollatorForSeq2Seq

# # we want to ignore tokenizer pad token in the loss
# label_pad_token_id = -100

# # Data collator
# data_collator = DataCollatorForSeq2Seq(
#     tokenizer,
#     model=model,
#     label_pad_token_id=label_pad_token_id,
#     pad_to_multiple_of=8
# )


# # In[36]:


# from huggingface_hub import HfFolder
# from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer


# # In[37]:


# batch_size = 96
# repository_id = f"{model_id.split('/')[1]}-finetuned-translation-10132023"
# training_args = Seq2SeqTrainingArguments(
#     output_dir=repository_id,
#     learning_rate=5e-5,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     generation_max_length=273,
#     weight_decay=0.01,
#     num_train_epochs=3,
#     predict_with_generate=True,
#     fp16=False,
#     bf16=True,
#     # logging & evaluation strategies
#     logging_dir=f"{repository_id}/logs",
#     logging_strategy="steps",
#     logging_steps=500,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     save_total_limit=2,
#     load_best_model_at_end=True,
#     # push to hub parameters
#     report_to="tensorboard",
#     hub_strategy="every_save",
#     hub_model_id=repository_id,
#     hub_token=HfFolder.get_token(),
#     push_to_hub=True,
# )


# # In[38]:


# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=trainset,
#     eval_dataset=testset,
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
# )


# # In[39]:


# trainer.train()


# # **Notes:**
# # 
# # batch_size = 128 -> OutOfMemoryError: CUDA out of memory.
# # 
# # batch_size = 96 -> OK
# # 
# # (needed to add the replacement of -100 for preds as well in compute metrics; otherwise get IndexError of "piece id is out of range")
# # 
# # 
# # each epoch ~13min (training 3min, evaluation ~10min)  
# # 3 epochs -> around 40min

# # In[40]:


# trainer.evaluate()


# # In[41]:


# repository_id


# # In[42]:


# # Save our tokenizer and create model card
# tokenizer.save_pretrained(repository_id)
# trainer.create_model_card()
# # Push the results to the hub
# trainer.push_to_hub()


# # In[ ]:





# # ## Use the fine-tuned model

# # In[14]:


# model_id = "google/flan-t5-xl" # Hugging Face Model Id
# repository_id = f"delmeng/{model_id.split('/')[1]}-finetuned-translation-10132023"
# tokenizer = T5Tokenizer.from_pretrained(repository_id)
# model = T5ForConditionalGeneration.from_pretrained(repository_id, device_map="auto")


# # In[15]:


# tokenizer.model_max_length


# # ### Still use the "translation" pipeline task type

# # In[16]:


# pipe_ft = pipeline("translation", model = repository_id, max_length=tokenizer.model_max_length, device_map="auto")


# # ### Evaluation on a small subset of the training set

# # In[19]:


# sample_size = 96
# train_ds_test = datasets.Dataset.from_pandas(training_features.head(sample_size), split="train")
# train_ds_test


# # In[20]:


# predicted_labels = []
# prediction = pd.DataFrame({"ID": pd.Series(dtype="int"),
#                    "predicted_label": pd.Series(dtype="str")})
# batch_size = 48
# # default batch size is 1, if not specified
# # with higher batch size, it's easier to trigger out of memory error

# for out in tqdm.tqdm(pipe_ft(KeyDataset(train_ds_test, "prompt"), batch_size=batch_size),total=len(train_ds_test)):
# # for out in pipe(KeyDataset(train_ds_raw, "prompt")):
# # for out in tqdm.tqdm(pipe(KeyDataset(train_ds_raw, "prompt"))):

#     #print(out)
#     generated_text = out[0]['translation_text']
#     predicted_labels.append(generated_text)


# # In[21]:


# prediction["ID"] = training_features.iloc[0:sample_size]["ID"]
# prediction["predicted_label"] = predicted_labels


# # In[13]:


# def bleu_func(x, y):
#     chencherry = SmoothingFunction()
#     x_split = [x_entry.strip().split() for x_entry in x]
#     y_split = y.strip().split()
#     return sentence_bleu(x_split, y_split, smoothing_function=chencherry.method3)

# def bleu_custom(y_true, y_pred, groups):
#     joined = pd.concat([y_true, y_pred, groups], axis=1)
#     joined[BLEU] = joined.apply(lambda x: bleu_func([x[y_true.name]], x[y_pred.name]), axis=1)
#     values = [joined[joined[groups.name] == unique][BLEU].mean() for unique in unique_list]
#     print(f"Overall mean: {joined[BLEU].mean()}")
#     print(f"Different genders: {values}")
#     print(f"Final score: {joined[BLEU].mean() - np.fabs(values[0] - values[1])/2}")
#     return joined[BLEU].mean() - np.fabs(values[0] - values[1])/2


# # In[22]:


# bleu_custom(
#     training_features.iloc[0:sample_size]["label"], 
#     prediction["predicted_label"], 
#     training_features.iloc[0:sample_size]["gender"]
# )


# # ### Evaluation on test dataset using the fine-tuned model

# # In[18]:


# predicted_labels = []
# test_prediction = pd.DataFrame({"ID": pd.Series(dtype="int"), "label": pd.Series(dtype="str")})
# batch_size = 32
# # default batch size is 1, if not specified
# # with higher batch size, it's easier to trigger out of memory error

# for out in tqdm.tqdm(pipe_ft(KeyDataset(test_ds_raw, "prompt"), batch_size=batch_size),total=len(test_ds_raw)):
#     generated_text = out[0]['translation_text']
#     predicted_labels.append(generated_text)

# test_prediction["ID"] = test_features["ID"]
# test_prediction["label"] = predicted_labels
# test_prediction.to_csv("t5_xl_finetuned_translation_submission-10142023.csv", index = False, encoding='utf-8-sig')


# # **Notes:**
# # 
# # when batch size = 48, got OOM error at     51%|█████     | 1536/3000 [12:25<11:50,  2.06it/s]
# # 
# # when batch size = 32 -> OK (30min)
# # 
# # 
# # final score: 0.265392 (compare with 0.167 using the pretrained model without fine-tuning)

# # In[ ]:





# # ## Try the "text2text-generation" pipeline task type

# # In[16]:


# pipe_ft = pipeline("text2text-generation", model = repository_id, max_length=tokenizer.model_max_length, device_map="auto")


# # ### Evaluation on a small subset of the training set

# # In[18]:


# sample_size = 96
# train_ds_test = datasets.Dataset.from_pandas(training_features.head(sample_size), split="train")
# train_ds_test


# # In[19]:


# predicted_labels = []
# prediction = pd.DataFrame({"ID": pd.Series(dtype="int"),
#                    "predicted_label": pd.Series(dtype="str")})
# batch_size = 48
# # default batch size is 1, if not specified
# # with higher batch size, it's easier to trigger out of memory error

# for out in tqdm.tqdm(pipe_ft(KeyDataset(train_ds_test, "prompt"), batch_size=batch_size),total=len(train_ds_test)):
#     generated_text = out[0]['generated_text']
#     predicted_labels.append(generated_text)


# # In[20]:


# prediction["ID"] = training_features.iloc[0:sample_size]["ID"]
# prediction["predicted_label"] = predicted_labels


# # In[21]:


# bleu_custom(
#     training_features.iloc[0:sample_size]["label"], 
#     prediction["predicted_label"], 
#     training_features.iloc[0:sample_size]["gender"]
# )


# # ### Evaluation on the test set

# # In[22]:


# predicted_labels = []
# test_prediction = pd.DataFrame({"ID": pd.Series(dtype="int"), "label": pd.Series(dtype="str")})
# batch_size = 32
# # default batch size is 1, if not specified
# # with higher batch size, it's easier to trigger out of memory error

# for out in tqdm.tqdm(pipe_ft(KeyDataset(test_ds_raw, "prompt"), batch_size=batch_size),total=len(test_ds_raw)):
#     generated_text = out[0]['generated_text']
#     predicted_labels.append(generated_text)

# test_prediction["ID"] = test_features["ID"]
# test_prediction["label"] = predicted_labels
# test_prediction.to_csv("t5_xl_finetuned_text_submission-10142023.csv", index = False, encoding='utf-8-sig')


# # when batch size = 32 -> 15min
# # 
# # Final score: 0.245683

# # **Observation:** the performance of the "text2text-generation" task type is not as good as the "translation" task type used above, although the inference seems to be much faster.

# # In[ ]:





# # # Fine tuning with Deepspeed

# # Reference: 
# # 
# # https://www.philschmid.de/fine-tune-flan-t5-deepspeed
# # 
# # https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/configs/ds_flan_t5_z3_config_bf16.json
# # 
# # https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/scripts/run_seq2seq_deepspeed.py
# # 
# # Note that the Deepspeed script and configuration file used below are based on these references.

# # In[3]:


# from huggingface_hub import notebook_login

# notebook_login()


# # The command to fine tune the model with Deepspeed:
# # 
# # ```
# # deepspeed --num_gpus=8 scripts/run_seq2seq_deepspeed.py \
# #     --model_id google/flan-t5-xl \
# #     --repository_id delmeng/flan-t5-xl-finetuning-translation-ds \
# #     --dataset_path training_data \
# #     --epochs 3 \
# #     --per_device_train_batch_size 96 \
# #     --per_device_eval_batch_size 96 \
# #     --generation_max_length 273 \
# #     --lr 1e-4 \
# #     --deepspeed config/deepspeed_config.json
# # ```
# # 
# # (Note that it took around 3.5h for this training job.)

# # ## Use the fine-tuned model for inference

# # After the training, the model was uploaded to my Hugging Face repository, so I can download and use it.
# # 
# # https://huggingface.co/delmeng/flan-t5-xl-finetuning-translation-ds/tree/main

# # In[13]:


# repository_id = "delmeng/flan-t5-xl-finetuning-translation-ds"
# tokenizer = T5Tokenizer.from_pretrained(repository_id)
# model = T5ForConditionalGeneration.from_pretrained(repository_id, device_map="auto")


# # In[21]:


# tokenizer.model_max_length


# # In[14]:


# pipe_ft = pipeline("translation", model = repository_id, max_length=tokenizer.model_max_length, device_map="auto")


# # ### Evaluation on a small subset of the training set

# # In[16]:


# sample_size = 96
# train_ds_test = datasets.Dataset.from_pandas(training_features.head(sample_size), split="train")
# train_ds_test


# # In[ ]:


# predicted_labels = []
# prediction = pd.DataFrame({"ID": pd.Series(dtype="int"),
#                    "predicted_label": pd.Series(dtype="str")})
# batch_size = 8

# for out in tqdm.tqdm(pipe_ft(KeyDataset(train_ds_test, "prompt"), batch_size=batch_size),total=len(train_ds_test)):

#     generated_text = out[0]['translation_text']
#     predicted_labels.append(generated_text)


# # Tried different batch size here => batch = 8 is a good choice.
# # 
# # batch = 48: This step is super slow!! Give up!!  
# # batch = 16: This step is super slow!! Give up!!  
# # batch = 8: 3min 48s  100%|██████████| 96/96 [03:48<00:00,  2.38s/it]  
# # batch = 1: 4min 100%|██████████| 96/96 [04:03<00:00,  2.53s/it]  
# # 

# # In[18]:


# prediction["ID"] = training_features.iloc[0:sample_size]["ID"]
# prediction["predicted_label"] = predicted_labels


# # In[19]:


# bleu_custom(
#     training_features.iloc[0:sample_size]["label"], 
#     prediction["predicted_label"], 
#     training_features.iloc[0:sample_size]["gender"]
# )


# # ### Evaluation on the test set

# # In[ ]:


# # note: this didn't work, it stuck at 6%-ish and couldn't finish

# predicted_labels = []
# test_prediction = pd.DataFrame({"ID": pd.Series(dtype="int"), "label": pd.Series(dtype="str")})
# batch_size = 8

# for out in tqdm.tqdm(pipe_ft(KeyDataset(test_ds_raw, "prompt"), batch_size=batch_size),total=len(test_ds_raw)):
#     generated_text = out[0]['translation_text']
#     predicted_labels.append(generated_text)


# # In[16]:


# predicted_labels = []
# test_prediction = pd.DataFrame({"ID": pd.Series(dtype="int"), "label": pd.Series(dtype="str")})

# for input_text in tqdm.tqdm(KeyDataset(test_ds_raw, "prompt")):
#     generated_text = pipe_ft(input_text)[0]['translation_text']
#     predicted_labels.append(generated_text)

# test_prediction["ID"] = test_features["ID"]
# test_prediction["label"] = predicted_labels
# test_prediction.to_csv("t5_xl_finetuned_translation_ds-10142023.csv", index = False, encoding='utf-8-sig')


# # 
# # The inference took 2.5 hours.
# # 
# # Final score: 0.262809

# # **Observation:** fine-tuning with Deepspeed didn't help with the fine-tuning in my case. In fact, somehow it slowed down the process. This could be caused by some configuration issue? The overall translation performance is similar to without Deepspeed though.

# # In[ ]:




