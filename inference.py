import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
model = deepspeed.init_inference(
    model,
    mp_size=2,
    dtype=torch.half,
    injection_policy={LlamaDecoderLayer: ('self_attn.o_proj', 'mlp.up_proj')}
)

batch = tokenizer(
    "The primary use of LLaMA is research on large language models, including",
    return_tensors="pt", 
    add_special_tokens=False
)
batch = {k: v.cuda() for k, v in batch.items()}
generated = model.generate(batch["input_ids"], max_length=100)
print(tokenizer.decode(generated[0]))
