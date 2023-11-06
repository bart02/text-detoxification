from .base_model import Model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm


class Mistral(Model):
    def __init__(self, device='cuda'):
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", 
                                                     device_map="auto", torch_dtype=torch.float16)
        self.device = device
    
    def generate(self, inputs):
        gen = []
        for inp in tqdm(inputs):
            encodeds = self.tokenizer(inp, return_tensors="pt")
            model_inputs = encodeds.to(self.device)
            generated_ids = self.model.generate(**model_inputs, 
                                                max_new_tokens=100, 
                                                do_sample=True, 
                                                eos_token_id=[2, 13], 
                                                temperature=1, 
                                                repetition_penalty=1.2)
            decoded = self.tokenizer.batch_decode(generated_ids[:, model_inputs["input_ids"].shape[1]:], 
                                                  skip_special_tokens=True)
            gen.append(decoded[0])
        return gen
