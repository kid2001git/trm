import transformers

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# nmt = pipeline('translation', model='../banglat5_nmt_en_bn')
# input_string = 'I go to school.'
# result = nmt(input_string)
# print('result:', result)

t = AutoTokenizer.from_pretrained("../bert-base-multilingual-cased")
m = AutoModelForSeq2SeqLM.from_pretrained("../bert-base-multilingual-cased")

nmt = pipeline('translation', model='m', tokenizer=t, src_lang='en', tgt_lang='fr')
result = nmt("I go to school.")
print('result:', result)
