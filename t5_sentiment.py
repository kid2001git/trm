from torchtext.models import T5Transform, T5_BASE_GENERATION
from torchtext.prototype.generate import GenerationUtils

# Text preprocessing pipeline
padding_idx = 0
eos_idx = 1
max_seq_len = 512
t5_sp_model_path = "https://download.pytorch.org/models/text/t5_tokenizer_base.model"
transform = T5Transform(sp_model_path=t5_sp_model_path, max_seq_len=max_seq_len, eos_idx=eos_idx, padding_idx=padding_idx)

# Model instantiation
t5_base = T5_BASE_GENERATION
transform = t5_base.transform()
model = t5_base.get_model()
model.eval()

# Generation
sequence_generator = GenerationUtils(model)
