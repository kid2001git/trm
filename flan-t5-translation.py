from transformers import T5Tokenizer, T5ForConditionalGeneration

path = './models/'

tokenizer = T5Tokenizer.from_pretrained(path + 'flan-t5-small')
model = T5ForConditionalGeneration.from_pretrained(path + 'flan-t5-small')

input_text = "translate English to Chinese: I go to school to learn."

input_ids = tokenizer(input_text, return_tensors="pt").input_ids
outputs = model.generate(input_ids)

translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(translated_text)
