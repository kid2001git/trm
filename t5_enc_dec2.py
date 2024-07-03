from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("../t5-base")
model = T5ForConditionalGeneration.from_pretrained("../t5-base")

task_prefix = "translate English to German: "
# use different length sentences to test batching
sentences = ["The house is wonderful.", "I like to work in NYC.","We go to school."]

inputs = tokenizer([task_prefix + sentence for sentence in sentences], return_tensors="pt", padding=True)

output_sequences = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    do_sample=False,  # disable sampling to test if batching affects output
)

print(tokenizer.batch_decode(output_sequences, skip_special_tokens=True))