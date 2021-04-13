# build the train val test datasets
import os
import json
from shutil import copyfile

cwd = os.getcwd()
xray_path = os.path.join(cwd, 'iu_xray')
image_path = os.path.join(xray_path, 'iu_xray_images')
caption_file = os.path.join(xray_path, 'iu_xray_captions.json')
with open(caption_file) as json_file:
    caption_dict = json.load(json_file)

# create data folder
dataset_path = os.path.join(cwd, 'datasets')
for split in ['train', 'valid', 'test']:
    split_path = os.path.join(dataset_path, split)
    if not os.path.exists(split_path):
        os.makedirs(split_path)

num_train = 5200
num_val = 700

_, _, filename = next(os.walk(image_path))

counter = 0
for item in filename:
    try:
        caption = caption_dict[item]
    except Exception:
        # no caption exists for this image, ignore it
        continue
    
    if counter < num_train: 
        # add to train folder
        split = 'train'
        index = str(counter)
    elif counter < num_train + num_val:
        split = 'valid'
        index = str(counter - num_train)
    else:
        split = 'test'
        index = str(counter - num_train - num_val)

    split_path = os.path.join(dataset_path, split)
    folder = os.path.join(split_path, str(index))
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # copy the image files and its caption into the folder
    src = os.path.join(image_path, item)
    tgt_image = os.path.join(folder, 'image.png')
    copyfile(src, tgt_image)
    # write caption of the corresponding image to the folder
    with open(os.path.join(folder, 'caption.txt'), 'w') as f:
        f.write(caption)
    
    counter += 1

