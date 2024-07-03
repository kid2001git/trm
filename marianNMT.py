import torch
from torch.utils.data import Dataset, DataLoader
from transformers import MarianMTModel, MarianTokenizer
import numpy as np

class TranslationDataset(Dataset):
    def __init__(self, dataset_path):
        self.hypotheses_cols_path = dataset_path + '/deen_nt2021_bleurt_0p2/hypotheses_cols.tsv'
        self.hypotheses_rows_path = dataset_path + '/deen_nt2021_bleurt_0p2/hypotheses_rows.tsv'
        self.scores_path = dataset_path + '/deen_nt2021_bleurt_0p2/scores.npy'
        # self.tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-de-en')
        self.tokenizer = MarianTokenizer.from_pretrained('./opus-mt-de-en')
        self.hypotheses_cols = []
        self.hypotheses_rows = []
        self.scores = []
        self.load_data()

    def load_data(self):
        with open(self.hypotheses_cols_path, 'r', encoding='utf-8') as f:
            self.hypotheses_cols = f.read().splitlines()
        with open(self.hypotheses_rows_path, 'r', encoding='utf-8') as f:
            self.hypotheses_rows = f.read().splitlines()
        self.scores = np.load(self.scores_path)

    def __len__(self):
        return min(len(self.hypotheses_cols), len(self.hypotheses_rows), len(self.scores))

    def __getitem__(self, idx):
        source_text = self.hypotheses_cols[idx]
        target_text = self.hypotheses_rows[idx]
        score = self.scores[idx]
        source_inputs = self.tokenizer.encode(source_text, padding='max_length', truncation=True, max_length=128,
                                              return_tensors='pt')
        target_inputs = self.tokenizer.encode(target_text, padding='max_length', truncation=True, max_length=128,
                                              return_tensors='pt')
        return {
            'source_inputs': source_inputs.squeeze(),
            'target_inputs': target_inputs.squeeze(),
            'score': score
        }


def collate_fn(batch):
    source_inputs = torch.stack([item['source_inputs'] for item in batch])
    target_inputs = torch.stack([item['target_inputs'] for item in batch])
    scores = torch.tensor([item['score'] for item in batch])
    return {
        'source_inputs': source_inputs,
        'target_inputs': target_inputs,
        'score': scores
    }

# dataset_path = '/kaggle/input/machine-translation-mbr-with-neural-metrics/de-en/newstest2021' 
dataset_path = './de-en/newstest2021' 
dataset = TranslationDataset(dataset_path)
dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)

# Performing sanity check of the dataloader
for batch in dataloader:
    print(batch)
    break

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model_name = './opus-mt-de-en'
model = MarianMTModel.from_pretrained(model_name).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# # Training loop
# for epoch in range(1):
#     for step,batch in enumerate(dataloader):
#         source_inputs = batch['source_inputs'].to(device)
#         target_inputs = batch['target_inputs'].to(device)
#         scores = batch['score'].to(device)

#         optimizer.zero_grad()
#         outputs = model(source_inputs, decoder_input_ids=target_inputs, return_dict=True)
#         logits = outputs.logits.flatten()
        
#         # Reshape scores to match the size of logits
#         scores = scores.view(-1)

#         # Resize logits to match the size of scores
#         logits = logits[:scores.size(0)]

#         # Convert logits and scores to Float dtype
#         logits = logits.float()
#         scores = scores.float()

#         loss = torch.nn.functional.mse_loss(logits, scores)
#         loss.backward()
#         optimizer.step()
        
#         print("Step-{}, Loss-{}".format(step,loss.item()))

# Inference loop
# Define the German text
german_text = "Guten Tag!"
# Load the tokenizer
model_name = './opus-mt-de-en'

tokenizer = MarianTokenizer.from_pretrained(model_name)
# Tokenize the German text
inputs = tokenizer.encode(german_text, return_tensors='pt')
# Perform inference
outputs = model.generate(inputs.to(model.device))
# Decode the English translation
english_translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
# Print the translated text
print("German Text: ", german_text)
print("English Translation: ", english_translation)