# merge file
# import numpy as np
# import scipy.io as sio
# path = '../../spm-data/'

# with open(path + 'en-full-v0.1.txt', 'r', encoding='utf-8') as file_src:
#     with open(path + 'jp-full-v0.1.txt', 'r', encoding='utf-8') as file_tgt:
#         with open(path + 'corpus_whole.txt', 'w', encoding='utf-8') as file_corpus:
#             while True:
#                 line_src = file_src.readline()
#                 line_tgt = file_tgt.readline()
#                 file_corpus.write(line_src)
#                 file_corpus.write('\t')
#                 file_corpus.write(line_tgt)
#                 if not line_src or not line_tgt:
#                     break
# file_src.close()
# file_tgt.close()
# file_corpus.close()

# from datasets import Dataset
# from datasets import load_dataset

# num_train = 1200000
# num_valid = 90000
# num_test = 10000

# en_jp_df_train = en_jp_df.iloc[:num_train]
# en_jp_df_valid = en_jp_df.iloc[num_train:num_train+num_valid]
# en_jp_df_test = en_jp_df.iloc[-num_test:]

# en_jp_df_train.to_csv("train.tsv", sep='\t', index=False)
# en_jp_df_valid.to_csv("valid.tsv", sep='\t', index=False)
# en_jp_df_test.to_csv("test.tsv", sep='\t', index=False)

# import evaluate
# my_metric = evaluate.load('../evaluate-main/metrics/sacreblue')
# print(my_metric)

