from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch

class BLIP2Pipeline:
    def __init__(self, model_id="Salesforce/blip2-opt-2.7b", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Blip2Processor.from_pretrained(model_id)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16 if "cuda" in self.device else torch.float32)
        self.model.to(self.device)

    def caption_image(self, image: Image.Image, max_tokens: int = 50) -> str:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.processor.decode(generated_ids[0], skip_special_tokens=True)

    def vqa(self, image: Image.Image, question: str, max_tokens: int = 50) -> str:
        inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.processor.decode(generated_ids[0], skip_special_tokens=True)
