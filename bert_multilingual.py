from transformers import pipeline
unmasker = pipeline('fill-mask', model='../bert-base-multilingual-cased')
result = unmasker("Hello I'm a [MASK] model.")
print('result:', result)
