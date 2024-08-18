import os
import json

# Paths
data_dir = 'data'
images_dir = os.path.join(data_dir, 'images')
train_images_dir = os.path.join(images_dir, 'train')
test_images_dir = os.path.join(images_dir, 'test')
llava_captions_path = os.path.join(data_dir, 'llava_captions_3.json')

# Labels map
label2id = {
        'Cultural_Religious': 0,
        'Fauna_Flora': 1,
        'Gastronomy': 2,
        'Nature': 3,
        'Sports': 4,
        'Urban_Rural': 5
}

# Vars definition
train_data = {}
test_data = {}

# Load captions
with open(llava_captions_path, 'r') as f:
    captions = json.load(f)
constant_prompt = "ER:  \nWhat is shown in this image? ASSISTANT: "

# Prepare train data
counter_not_found = 0
for dir in os.listdir(train_images_dir):
    for file in os.listdir(os.path.join(train_images_dir, dir)):
        if file.endswith('.jpg'):
            try:
                train_data[file] = {'caption': captions[file].replace(constant_prompt, ''), 'label': label2id[dir]}
            except Exception as e:
                counter_not_found += 1
                print(f'Could not find {file} image in train')

print(f'Could not find caption for {counter_not_found} images in train')

with open(os.path.join(data_dir,'train_data.json'), 'w') as f:
    json.dump(train_data, f, indent=4)

# Prepare test data
counter_not_found = 0
for dir in os.listdir(test_images_dir):
    for file in os.listdir(os.path.join(test_images_dir, dir)):
        if file.endswith('.jpg'):
            try:
                test_data[file] = {'caption': captions[file].replace(constant_prompt, ''), 'label': label2id[dir]}
            except Exception as e:
                counter_not_found += 1
                print(f'Could not find {file} image in test')

print(f'Could not find caption for {counter_not_found} images in test')

with open(os.path.join(data_dir,'test_data.json'), 'w') as f:
    json.dump(test_data, f, indent=4)
    
# Prepare full data
train_data.update(test_data)
with open(os.path.join(data_dir, 'full_data.json'), 'w') as f:
    json.dump(train_data, f, indent=4)

