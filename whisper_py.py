from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset


# load model and processor
processor = WhisperProcessor.from_pretrained("../whisper/whisper_en_base")
model = WhisperForConditionalGeneration.from_pretrained("../whisper/whisper_en_base")

# load dummy dataset and read audio files
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

# generate token ids
predicted_ids = model.generate(input_features)
# decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
