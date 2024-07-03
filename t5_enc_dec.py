from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("../t5-base")
model = T5ForConditionalGeneration.from_pretrained("../t5-base")

input_ids = tokenizer("translate English to Chinese: The house is wonderful.", return_tensors="pt").input_ids
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))