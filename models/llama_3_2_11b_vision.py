import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

class Llama32VisionModel:
    def __init__(self, model_id="meta-llama/Llama-3.2-11B-Vision-Instruct"):
        self.model_id = model_id
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def process_image_and_text(self, image_url, text_prompt):
        image = Image.open(requests.get(image_url, stream=True).raw)
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": text_prompt}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(image, input_text, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=30)
        return self.processor.decode(output[0])

if __name__ == "__main__":
    model = Llama32VisionModel()
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
    text_prompt = "If I had to write a haiku for this one, it would be: "
    result = model.process_image_and_text(image_url, text_prompt)
    print(result)
