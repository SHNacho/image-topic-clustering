import json
import os
import torch
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

def generate_caption(
        image_path: str, 
        model, 
        processor, 
        device: str):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    out = model.generate(**inputs)
    caption = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    return caption

def generate_captions(image_dir: str, device: str):
    captions = dict()
    model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", 
            #device_map={"": 0}, 
            #torch_dtype=torch.float16
        ).to(device)
    processor = Blip2Processor.from_pretrained(
            "Salesforce/blip2-opt-2.7b")

    for image_file in tqdm(os.listdir(image_dir)[:20]):
        image_path = os.path.join(image_dir, image_file)
        caption = generate_caption(image_path, model, processor, device)
        captions[image_file] = caption

    return captions

def save_captions(captions: dict, save_path: str):
    with open(save_path, 'w') as f:
        json.dump(captions, f, indent=4)
    print(f"Captions saved to {save_path}")

if __name__ == "__main__":
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    captions = generate_captions("data/images", device)
    save_captions(captions, "data/captions_blip2.json")



