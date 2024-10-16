import json
import os
import torch
from tqdm import tqdm
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration, 
    BlipForQuestionAnswering
)
from PIL import Image

def generate_caption(
        image_path: str, 
        model, 
        processor, 
        device: str):
    image = Image.open(image_path).convert("RGB")
    prompt = "What is the general topic for this image?"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def generate_captions(image_dir: str, device: str):
    captions = dict()
    model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-capfilt-large").to(device)
    processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-vqa-capfilt-large")

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    captions = generate_captions("data/images", device)
    save_captions(captions, "data/captions_blip_vqa_2.json")


