from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("../bert-base-uncased")
# sents = [
#     '选择珠江花园的原因就是方便。',
#     '笔记本的键盘确实爽。',
#     '房间太小。其他的都一般。',
#     '今天才知道这书还有第6卷,真有点郁闷.',
#     '机器背面似乎被撕了张什么标签，残胶还在。',
# ]
sents =[
    'Hello, world',
    'I go to school'
]

#编码两个句子
out = tokenizer.encode(
    text=sents[0],
    text_pair=sents[1],  # 一次编码两个句子，若没有text_pair这个参数，就一次编码一个句子

    #当句子长度大于max_length时,截断
    truncation=True,

    #一律补pad到max_length长度
    padding='max_length',   # 少于max_length时就padding
    add_special_tokens=True,
    max_length=30,
    return_tensors=None,  # None表示不指定数据类型，默认返回list
)

print(out)

print(tokenizer.decode(out))

# from huggingface_hub import hf_hub_download

# hf_hub_download(repo_id="bigscience/T0_3B", filename="config.json", cache_dir="./bigscience_t0")