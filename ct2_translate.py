import ctranslate2
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("../converted/source.spm")

source = sp.encode("Hello world!", out_type=str)

translator = ctranslate2.Translator("../converted")

results = translator.translate_batch([source])

output = sp.decode(results[0].hypotheses[0])
print('\n', output)