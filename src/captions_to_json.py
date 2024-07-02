import json
import os
from tqdm import tqdm

captions = dict()
captions_dir = 'data/captions'
for file in tqdm(os.listdir('data/captions')):
    file_path = os.path.join(captions_dir, file)
    with open(file_path, 'r') as f:
        caption = f.readline()
        key = file.replace('.txt', '')
        captions[key] = caption

with open('data/captions.json', 'w') as f:
    json.dump(captions, f, indent=4)

