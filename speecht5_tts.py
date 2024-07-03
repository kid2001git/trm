# from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech

# path = './models/'
# processor = SpeechT5Processor.from_pretrained(path + "speecht5_tts")
# model = SpeechT5ForTextToSpeech.from_pretrained(path + "speecht5_tts")

# inputs = processor(text="Don't count the days, make the days count.", return_tensors="pt")

# print(inputs)

# from datasets import load_dataset
# embeddings_dataset = load_dataset("./cmu-arctic-xvectors", split="validation")

# import torch
# speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)
# print(spectrogram)

# from transformers import SpeechT5HifiGan
# vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
# print(vocoder)


# with torch.no_grad():
#     speech = vocoder(spectrogram)

# speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

# import soundfile as sf
# sf.write("tts_example.wav", speech.numpy(), samplerate=16000)

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset

path = './models/'

processor = SpeechT5Processor.from_pretrained(path + "speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained(path + "speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained(path + "speecht5_hifigan")

# inputs = processor(text="Hello, my dog is cute.", return_tensors="pt")
# inputs = processor(text="Hello, we are the students of university.", return_tensors="pt")
inputs = processor(text="Recalling also the outcome of the twenty-third special session of the General Assembly, entitled Women 2000: gender equality, development and peace for the twenty-first century,Resolution S-23/2, annex, and resolution S-23/3, annex. and the proposed actions and initiatives to overcome obstacles and challenges thereto",
                        return_tensors="pt")

print(inputs)
# load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

sf.write("speech.wav", speech.numpy(), samplerate=16000)




