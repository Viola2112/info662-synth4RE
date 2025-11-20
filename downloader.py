from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json
import os
import gc

#save_dir = '/data/user/rrcron/synth_gen_clone/Qwen3-32B'
#save_dir = '/data/user/rrcron/synth_gen_clone/gemma-3-27b-it'
#save_dir = '/data/user/rrcron/synth_gen_clone/medgemma-27b-text-it'
save_dir = './Qwen3-8B'

model_name = "Qwen/Qwen3-8B"
#model_name = "google/gemma-3-27b-it"
#model_name = "google/medgemma-27b-text-it"
#hf_token = json.load(open('hconfig.json', 'r'))['hf_token']

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
#     )

#os.mkdir(save_dir)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto')
print(f"Model downloaded. Saving to {save_dir}")
model.save_pretrained(f'{save_dir}/')
del model
gc.collect()

tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Tokenizer downloaded. Saving to {save_dir}")
tokenizer.save_pretrained(f'{save_dir}/')
del tokenizer
gc.collect()