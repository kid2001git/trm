from transformers import pipeline

model_path = '../'


summarizer = pipeline('summarization', model_path + 'summarization')
text = """
This chapter is the first of two chapters that discusses the transformer architecture, which is the foundation for a plethora of language models, such as BERT and its variants (discussed in Chapters 4 and 5), as well as the GPT-x family from OpenAI, various other LLMs (discussed in Chapters 6 and 7). This chapter describes the main components of the original transformer, along with Python-based transformer code samples for various NLP tasks, such as NER, QnA, and mask filling tasks. Please read chapter one that discusses some topics that are relevant to the material in this chapter. The first part of this chapter introduces the transformer architecture that was developed by Google and released in late 2017. This section also discusses some NLP Transformer models. The second part of this chapter discusses the transformers library from Hugging Face, which is a company that provides a repository of more than 20,000 transformer-based models. This section also contains Python-based transformer code samples that perform various NLP tasks, such as NER, QnA, and mask filling tasks. The third part of this chapter provides more details regarding the encoder component, along with additional details regarding the attention mechanism, as well as the decoder component, which is the other main component of the transformer architecture.  Before you read this chapter, keep in mind that there are cases of "forward referencing" concepts that are discussed in later chapters, which provide context for a given concept. Of course, if you are unfamiliar with a concept discussed in a later chapter, then it will also be challenging to fully understand the topic under discussion. As a result, you will probably need to read some topics more than once, after which concepts become easier to grasp when you understand the current context and the context of referenced material in later chapters (fortunately, this reading process is finite).
"""
result = summarizer(text)
print("summary text:",result)
print("summary text0:",result[0]['summary_text'])
# summary text0: This chapter is the first of two chapters that discusses the transformer architecture . This chapter describes the main components of the original transformer, along with Python-based transformer code samples for various NLP tasks, such as NER, QnA, and mask filling tasks .
