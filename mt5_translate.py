from transformers import MT5ForConditionalGeneration, T5Tokenizer

print('0')
# model_path = './models/'
model_path = './models/'
model = MT5ForConditionalGeneration.from_pretrained(model_path+'mt5-small')
print('0.5')
tokenizer = T5Tokenizer.from_pretrained(model_path+'mt5-small')

print('1')

text = "Your text to be translated"
inputs = tokenizer.encode("translate English to French: "+ text, return_tensors='pt')

print('2')
print('inputs', inputs)
outputs = model.generate(inputs, max_length=40, num_beams=1, early_stopping=True)
print('outputs:',outputs)
print(outputs[0])
translation = tokenizer.decode(outputs[0])
print(translation)
print('3')


