import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder.lstm import LSTMDecoder
from encoder.encoder import EncoderCNN

class EncoderDecoderModel(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.encoder = EncoderCNN()
        self.decoder = LSTMDecoder(tokenizer)

    def forward(self, image, caption, **kwargs):
        _, encoder_out = self.encoder(image, **kwargs)
        decoder_out = self.decoder(caption, encoder_out, **kwargs)
        return decoder_out
        
if __name__ == '__main__':
    # test encoder decoder 
    import sys, os
    sys.path.append(os.path.abspath(os.path.join('..', '')))
    from utils.tokenizer import Tokenizer
    from utils.dataset import ChestXrayDataSet, collate_fn
    from torchvision import transforms

    caption_dir = '/mnt/d/Github/MICaptioning/iu_xray/iu_xray_captions.json'
    data_dir = '/mnt/d/Github/MICaptioning/datasets'
    tokenizer = Tokenizer(caption_dir)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    train_dataset = ChestXrayDataSet(data_dir, 'train', tokenizer, transform)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=4,
                                                   shuffle=False,
                                                   collate_fn=collate_fn)

    encoder_decoder = EncoderDecoderModel(tokenizer)
    # print(encoder_decoder)
    for img, caption in train_dataloader:
        # test encoding img into features
        out = encoder_decoder(img, caption)
        print(out.shape)
        break