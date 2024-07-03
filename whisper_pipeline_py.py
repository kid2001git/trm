from transformers import pipeline
import whisper

# whisper_model = pipeline('automatic-speech-recognition', model='../whisper/whisper_en_base')
# testfile = '../hydrogen.mp3'
# result  = whisper_model(testfile)
# print(result)

model = whisper.load_model("base")
result = model.transcribe("../hydrogen.mp3")
print(result)
