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
    def __init__(self, choice, tokenizer):
        super().__init__()
        self.encoder = EncoderCNN(choice)
        self.decoder = LSTMDecoder(tokenizer)
        self.tokenizer = tokenizer

    def forward(self, image, caption, **kwargs):
        _, encoder_out = self.encoder(image, **kwargs)
        decoder_out = self.decoder(caption, encoder_out, **kwargs)
        return decoder_out

    def inference(self, image):
        _, ip = self.encoder(image) # avg_features (bz x 512)
        max_len = 100
        hidden = None
        ids_list = []
        is_decoding = torch.ones(ip.size(0)).bool()
        for t in range(max_len):
            lstm_out, hidden = self.decoder.lstm(ip.unsqueeze(1), hidden)
            linear_out = self.decoder.linear(lstm_out).squeeze(1)
            _, max_ids = linear_out.max(dim=1)
            max_ids.masked_fill_(
                ~is_decoding,
                self.tokenizer.pad
            )
            ids_list.append(max_ids)
            is_decoding = is_decoding * torch.ne(max_ids, self.tokenizer.eos)
            if (torch.all(~is_decoding)):
                break
            ip = self.decoder.embed(max_ids)
        ids_list = torch.transpose(torch.stack(ids_list), 0, 1)
        return ids_list

        # BLEU: 15.18

    def beam_search(self, beam_size, device, SingleBeamSearchBoard, image, n_best=1):
        _, encoder_out = self.encoder(image)
        bz, hidden_size = encoder_out.shape
        max_len = 100
        # initial input is bos
        input_token = encoder_out.new_full(
            (bz, 1),
            self.tokenizer.bos
        ).long().squeeze()
        # final output
        output = encoder_out.new_full(
            (bz, max_len),
            self.tokenizer.pad
        ).long()
        # init beam search boards, one for each batch
        prev_hidden = encoder_out.new_full(
            (1, 1, hidden_size),
            0
        )
        boards = [SingleBeamSearchBoard(
            device,
            self.tokenizer,
            # prev_status_config
            {
                "prev_hidden": prev_hidden, # 1x1xhidden_size
                "prev_cell": encoder_out.new_full(
                    (1, 1, 512),
                    0
                )
            }
        ) for i in range(bz)]

        for i in range(max_len):
            # all batch has finished beam search
            if sum([boards[t].is_done() for t in range(bz)]) == bz:
                break
            for j in range(bz):
                # perform beam search per batch
                board = boards[j]
                # sample size = beam size, decode for one step
                # prev_y = beam_size x 1, regard beam_size as batch size here
                # embed(prev_y): beam_size x 1 x embed_dim
                prev_y, prev_status = board.get_batch() 
                prev_hidden = prev_status['prev_hidden'] # 1 x beam_size x hidden_size
                prev_cell = prev_status['prev_cell'] # 1 x beam_size x hidden_size
                if i == 0:
                    x, (new_hidden, new_cell) = self.decoder.lstm(encoder_out[j].unsqueeze(0).unsqueeze(1))
                else:
                    x, (new_hidden, new_cell) = self.decoder.lstm(self.decoder.embed(prev_y), (prev_hidden, prev_cell))
                logit = self.decoder.linear(x)
                logit = nn.LogSoftmax(dim=2)(logit)
                # crucial step, update the beam board with cumulative prob
                board.collect_result(logit, {
                    "prev_hidden": new_hidden,
                    "prev_cell": new_cell
                })
        # finished beam search, generate the token
        output_sent = []
        output_prob = []
        for board in boards:
            sentence, probs = board.get_n_best(n_best)
            output_sent.append(sentence)
            output_prob.append(probs)
        return output_sent, output_prob
            
  
if __name__ == '__main__':
    # test encoder decoder 
    import sys, os
    sys.path.append(os.path.abspath(os.path.join('/mnt/d/Github/MICaptioning', '')))
    from utils.tokenizer import Tokenizer
    from utils.dataset import ChestXrayDataSet, collate_fn
    from torchvision import transforms
    from utils.search import SingleBeamSearchBoard

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

    encoder_decoder = EncoderDecoderModel('vgg', tokenizer)
    # print(encoder_decoder)
    for img, caption, tags_vec in train_dataloader:
        # test encoding img into features
        # out = encoder_decoder.inference(img)
        # out = encoder_decoder(img, caption)
        sent, prob = encoder_decoder.beam_search(5, torch.device('cpu'), SingleBeamSearchBoard, img)
        print(sent)
        # print(prob)
        break