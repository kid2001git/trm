from transformers import BertTokenizerFast, BertModel   

tokenizer_bert = BertTokenizerFast.from_pretrained("../bert-base", local_files_only=True)
model_bert = BertModel.from_pretrained("../bert-base", local_files_only=True)
print('load suc')
input_data = 'We go to school'
inputs = tokenizer_bert(input_data, padding=True, truncation=True, return_tensors="pt")
print(inputs)
tokens = tokenizer_bert.tokenize(input_data)
print(tokens)
ids = tokenizer_bert.convert_tokens_to_ids(tokens)
print(ids)
decoded_string = tokenizer_bert.convert_ids_to_tokens(ids)
print(decoded_string)
print('end')