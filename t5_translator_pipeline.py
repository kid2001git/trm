# from transformers import pipeline
# from pinferencia import Server

# t5 = pipeline(model="../t5-base", tokenizer="t5-base")


# def translate(text):
#     return t5(text)

# service = Server()
# service.register(model_name="t5", model=translate)

# #err

from transformers import T5Tokenizer, T5ForConditionalGeneration
ckpt = './models/kj-t5-base'
tokenizer = T5Tokenizer.from_pretrained(ckpt)

model = T5ForConditionalGeneration.from_pretrained(ckpt, return_dict=True)

input = "My name is Azeem and I live in India"

# You can also use "translate English to French" and "translate English to Romanian"
input_ids = tokenizer("translate English to Japanese: "+input, return_tensors="pt").input_ids  # Batch size 1

outputs = model.generate(input_ids)

decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(decoded)