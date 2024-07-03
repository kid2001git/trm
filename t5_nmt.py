from torchtext.models import T5Transform

padding_idx = 0
eos_idx = 1
max_seq_len = 512
t5_sp_model_path = "path_to_your_sentencepiece_model"

transform = T5Transform(
    sp_model_path=t5_sp_model_path,
    max_seq_len=max_seq_len,
    eos_idx=eos_idx,
    padding_idx=padding_idx,
)

from torchtext.models import T5_BASE_GENERATION

t5_base = T5_BASE_GENERATION
transform = t5_base.transform()
model = t5_base.get_model()
model.eval()

from torchtext.prototype.generate import GenerationUtils

sequence_generator = GenerationUtils(model)

