from transformers import T5Tokenizer, T5Model

tokenizer_t5 = T5Tokenizer.from_pretrained("../t5-base")
model_t5 = T5Model.from_pretrained("../t5-base")

# input_ids = tokenizer(
#     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
# ).input_ids  # Batch size 1
# decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

# # forward pass
# outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
# last_hidden_states = outputs.last_hidden_state
input_data = 'Wearegoingtoschool.'
tokens = tokenizer_t5.tokenize(input_data)
print(tokens)
ids = tokenizer_t5.convert_tokens_to_ids(tokens)
print(ids)
decoded_string = tokenizer_t5.convert_ids_to_tokens(ids)
print(decoded_string)
print('end')

