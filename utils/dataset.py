# Load datasets, also create vocab for tokenizer
import torch
from torch.utils.data import Dataset
from skimage import io
import os
import json
import numpy as np
from torchvision import transforms
import pickle
from utils.tokenizer import Tokenizer
from utils.tag_utils import tags2array


class ChestXrayDataSet(Dataset):
    def __init__(self,
                 input_dir,
                 op,
                 tokenizer,
                 transform=None):
        self.op = op
        try:
            if self.op == 'train':
                self.data_dir = os.path.join(input_dir, 'train')
            elif self.op == 'valid':
                self.data_dir = os.path.join(input_dir, 'valid')
            elif self.op == 'test':
                self.data_dir = os.path.join(input_dir, 'test')
            else:
                raise ValueError
        except ValueError:
            print('op should be either train, val or test!')
        self.tokenizer = tokenizer
        if not transform:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform


    def __getitem__(self, index):
        img = io.imread(os.path.join(self.data_dir, str(index), 'image.png'))
        with open(os.path.join(self.data_dir, str(index), 'caption.txt'), 'r') as f:
            caption = f.read()
        with open(os.path.join(self.data_dir, str(index), 'autotags.txt'), 'r') as f:
            autotags = f.read()
        
        out_img = self.transform(img)
        tags_vec = np.array(tags2array(autotags))
        # sample = {'img': out_img[0, :, :].unsqueeze(0), 'caption': caption}
        # 1 channel img: return out_img[0, :, :].unsqueeze(0), self.tokenizer.encode(caption), self.tokenizer.pad
        return out_img, self.tokenizer.encode(caption), self.tokenizer.pad, tags_vec

    def __len__(self):
        return len(next(os.walk(self.data_dir))[1])


# need collate function to pad the sentences
def collate_fn(data):
    images, captions, pads, tags_vec = zip(*data)
    pad_index = pads[0]
    image = torch.stack(images, 0)
    bz = len(captions)
    maxlen = max([len(sent) for sent in captions])
    caption = image.new_full(
        (bz, maxlen),
        0
    )
    for i in range(bz):
        pad_len = maxlen - len(captions[i])
        padding = torch.ones(pad_len) * pad_index
        token = torch.tensor(captions[i])
        caption[i, :] = torch.cat((token, padding), dim=0)

    return image, caption.long(), tags_vec


def get_loader(input_dir,
               op,
               tokenizer,
               transform,
               batch_size,
               shuffle=False):
    dataset = ChestXrayDataSet(input_dir=input_dir,
                               op=op,
                               tokenizer=tokenizer,
                               transform=transform
                               )

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_fn)
    return data_loader


if __name__ == '__main__':
    # test data loader
    input_dir = '/mnt/d/Github/MICaptioning/datasets'
    op = 'test'
    batch_size = 2
    resize = 256
    crop_size = 224
    transform = transforms.Compose([
        transforms.ToTensor(),
        # resize is necessary!
        transforms.Resize(resize),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    
    caption_dir = '/mnt/d/Github/MICaptioning/iu_xray/iu_xray_captions.json'
    tokenizer = Tokenizer(caption_dir)

    dataloader = get_loader(input_dir, op, tokenizer, transform, batch_size)
    import matplotlib.pyplot as plt
    for item in dataloader:
        img, caption = item
        print(img.shape)
        print(caption)
        break
 

 


