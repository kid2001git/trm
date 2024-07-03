from transformers import pipeline

sentiment_model = pipeline('sentiment-analysis',model="../bert-base-cased-sentiment")
input_string = ['I like', 'I don't like']
# print(len(input_string))
for i in range (len(input_string)):
    result = sentiment_model(input_string[i])
    print(print(input_string[i], result))

# result = sentiment_model(input_string[1])
# print(input_string[1], result)

