__author__ = "Skye Lane Goetz"


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class TransformersLLM:

    def __init__(self, model_name: str = "microsoft/phi-2"):
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map=None
        ).to(self.device)

    def invoke(self, prompt: str) -> str:
        hyperparameters = {
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": 0.3,  # Low randomness for structured generation
            "top_p": 0.6,  # Focuses on the most likely outputs, low creativity
            "repetition_penalty": 1.2,  # Prevents loops in structured blocks
        }
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, **hyperparameters)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
