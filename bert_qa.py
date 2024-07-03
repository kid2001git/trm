from transformers import pipeline

qa_model = pipeline('question-answering', model='../bert-base-multilingual-cased')
question='Who are you?'
context = 'I was born in Italy. I am a boy. I like film'
result = qa_model({'question':question, 'context':context})
print(result)