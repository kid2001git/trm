from transformers import BertTokenizer, BertModel

tokenizer_multi = BertTokenizer.from_pretrained('../bert-base-multilingual-cased')
model = BertModel.from_pretrained("../bert-base-multilingual-cased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer_multi(text, return_tensors='pt')
output = model(**encoded_input)
print('encoded_input:',encoded_input)
# print('output:',output)
input_data = '아버지가방으로들어간다'
tokens = tokenizer_multi.tokenize(input_data)
print('tokens:',tokens)
input_ids = tokenizer_multi.convert_tokens_to_ids(tokens)
print(input_ids)
decoded_string = tokenizer_multi.convert_ids_to_tokens(input_ids)
print('decoded_string:',decoded_string)
print('end')