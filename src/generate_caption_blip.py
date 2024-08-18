import json
import os
import torch
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

def generate_caption(
        image_path: str, 
        model, 
        processor, 
        device: str):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def generate_captions(image_dir: str, model, processor):
    captions = dict()

    for image_file in tqdm(os.listdir(image_dir)):
        if image_file.endswith(".jpg"):
            image_path = os.path.join(image_dir, image_file)
            caption = generate_caption(image_path, model, processor, device)
            captions[image_file] = caption

    return captions

def save_captions(captions: dict, save_path: str):
    with open(save_path, 'w') as f:
        json.dump(captions, f, indent=4)
    print(f"Captions saved to {save_path}")

if __name__ == "__main__":
    label2id = {
            'Cultural_Religious': 0,
            'Fauna_Flora': 1,
            'Gastronomy': 2,
            'Nature': 3,
            'Sports': 4,
            'Urban_Rural': 5
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large", 
        ).to(device)
    processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large")
    
    image_dir = "data/images"
    for dir in tqdm(os.listdir(image_dir)):
        data = dict()
        path = os.path.join(image_dir, dir)
        for class_dir in tqdm(os.listdir(path), desc=path):
            class_path = os.path.join(path, class_dir)
            class_captions = generate_captions(
                image_dir=class_path, 
                model=model, processor=processor
            )
            class_captions = {
                key: {'caption': value, 'label': label2id[class_dir]} for key, value in class_captions.items()
            }
            data.update(class_captions)
        save_captions(data, f"data/{dir}_captions_blip.json")

    




