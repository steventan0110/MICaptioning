import torch
import torch.nn as nn

class LSTMDecoder(nn.Module):
    def __init__(
        self, tokenizer, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1,
    ):
        """
            LSTM hidden state has dimension: (num_layers*num_directions, batch_size, hidden_size)
            Because the image feature encoded by VGG has shape (batch_size, 512 channels), hidden_size
            must match 512 => LSTM(embed_dim, 512, num_layers)
        """
        super().__init__()
        # model params defined here
        self.tokenizer = tokenizer
        self.num_layers = num_layers
        vocab_size = self.tokenizer.vocab_size
        # TODO: load pretrained embedding or use train from scratch
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, image_feature):
        embeddings = self.embed(x)
        input_token = embeddings.permute(1,0,2)
        bz, hidden_size = image_feature.size(0), image_feature.size(1)
        c0 = image_feature.new_full(
            (self.num_layers, bz, hidden_size),
            0)
        h0 = image_feature.unsqueeze(0) #  num_layer x batch_size x hidden_dim
        output, (state, _) = self.lstm(input_token, (h0, c0))
        # output: seq_len x batch x hidden_dim, state: num_layer x batch x hidden_dim
        logits = self.dropout(self.linear(output)) # seqlen x batch size x vocab size
        return logits.permute(1, 0, 2)


if __name__ == '__main__':
    # test decoder given encoder output
    import sys, os
    sys.path.append(os.path.abspath(os.path.join('../..', '')))
    from utils.tokenizer import Tokenizer
    from utils.dataset import ChestXrayDataSet, collate_fn
    from torchvision import transforms
    from models.encoder.encoder import EncoderCNN

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
    encoder = EncoderCNN()
    decoder = LSTMDecoder(tokenizer)
    # print(encoder)
    for img, caption in train_dataloader:
        # test encoding img into features
        _, encoder_out = encoder(img)
        decoder_out = decoder(caption, encoder_out)
        break
