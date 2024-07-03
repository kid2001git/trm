from transformers import pipeline

sentiment_model_multilingual = pipeline('sentiment-analysis', model='../bert-base-multilingual-cased')
input_string = 'I am good.'
result = sentiment_model_multilingual(input_string)
print('result:', result)

