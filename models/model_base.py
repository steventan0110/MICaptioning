import sys
sys.path.append('/Users/chenyuzhang/Desktop/JHU-6/DL/MICaptioning')

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.decoder.lstm import LSTMDecoder
from models.encoder.encoder import EncoderCNN
# used for local tesing:
# from decoder.lstm import LSTMDecoder
# from encoder.encoder import EncoderCNN

class EncoderDecoderModel(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.encoder = EncoderCNN()
        self.decoder = LSTMDecoder(tokenizer)
        self.tokenizer = tokenizer

    def forward(self, image, caption, **kwargs):
        _, encoder_out = self.encoder(image, **kwargs)
        decoder_out = self.decoder(caption, encoder_out, **kwargs)
        return decoder_out

    def inference(self, image):
        _, ip = self.encoder(image)
        max_len = 30
        hidden = None
        ids_list = []
        for t in range(max_len):
            lstm_out, hidden = self.decoder.lstm(ip.unsqueeze(1), hidden)
            linear_out = self.decoder.linear(lstm_out).squeeze(1)
            _, word_caption = linear_out.max(dim=1)
            ids_list.append(word_caption)
            ip = self.decoder.embed(word_caption)
        ids_list = torch.transpose(torch.stack(ids_list), 0, 1)
        return ids_list

        # _, encoder_out = self.encoder(image)
        # bz = encoder_out.size(0)
        # input_token = encoder_out.new_full(
        #     (bz, 1),
        #     self.tokenizer.bos
        # ).long().squeeze()
        # # should be replaced by an argument
        # max_len = 30
        # output = encoder_out.new_full(
        #     (bz, max_len),
        #     self.tokenizer.pad
        # ).long()
        # output[:, 0] = input_token
        # prev_hidden = encoder_out.unsqueeze(0)
        # prev_c = encoder_out.new_full(
        #     (1, bz, 512), #(self.num_layers, bz, hidden_size)
        #     0
        # )
        
        # is_decoding = encoder_out.new_ones(bz).bool()
        # for i in range(max_len-1):
        #     embed_token = self.decoder.embed(input_token) # bz x embed_dim
        #     input_token = embed_token.unsqueeze(dim=1)
        #     x, (prev_hidden, prev_c) = self.decoder.lstm(input_token, (prev_hidden, prev_c))
        #     logit = self.decoder.linear(x)
        #     # greedy search
        #     indice = logit.squeeze().argmax(dim=1) # bz
        #     new_token = indice.masked_fill_(
        #         ~is_decoding,
        #         self.tokenizer.pad
        #     )
        #     is_decoding = is_decoding * torch.ne(new_token, self.tokenizer.eos)
        #     input_token = new_token
        #     output[:,i+1] = new_token
        #     if torch.all(~is_decoding):
        #         # all batch are not decoding
        #         break
        # return output
        
if __name__ == '__main__':
    # test encoder decoder 
    import sys, os
    sys.path.append(os.path.abspath(os.path.join('..', '')))
    from utils.tokenizer import Tokenizer
    from utils.dataset import ChestXrayDataSet, collate_fn
    from torchvision import transforms

    # caption_dir = '/mnt/d/Github/MICaptioning/iu_xray/iu_xray_captions.json'
    # data_dir = '/mnt/d/Github/MICaptioning/datasets'
    caption_dir = '//Users/chenyuzhang/Desktop/JHU-6/DL/MICaptioning/iu_xray/iu_xray_captions.json'
    data_dir = '/Users/chenyuzhang/Desktop/JHU-6/DL/MICaptioning/datasets'
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
        out = encoder_decoder.inference(img)
        # out = encoder_decoder(img, caption)
        print(out.shape)
        break