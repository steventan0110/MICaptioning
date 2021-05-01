import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (self.head_dim*heads == embed_size)
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads* self.head_dim, self.embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0] # bz
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]  # src and target sent length
        # split into heads
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # query key multiplication
        # query: (N, query_len, heads, heads_dim)
        # key: (N, key_len,  heads, heads_dim)
        # energy/attention: (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) # so we don't need to write flatten and batch matrix multiplication
       
        if mask is not None: # mask: (N, 1, 1, query_len) for src, (N, 1, query_len, query_len) for tgt
            energy = energy.masked_fill(mask==0, float("-1e20")) # mask is broadcast
        
        attention = torch.softmax(energy/(self.embed_size**(1/2)), dim=3)

        # energy/attention: (N, heads, query_len, key_len)
        # value: (N, value_len, heads, heads_dim)
        # out: N, query_len, heads, head_dim => concat the heads
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.head_dim*self.heads
        ) 
        out = self.fc_out(out)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size), # some extra mapping to higher dimension
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        # skip connect
        x = self.dropout(self.norm1(attention+query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x)) # skip connect
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_len, # limit for positional embedding to work
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        # self.embed = nn.Embedding(src_vocab_size, embed_size) no need for image feature
        # self.position_embed = nn.Embedding(max_len, embed_size) already added
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                ) for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        N, seq_length, embed_size = x.shape
        # position = torch.arange(0, seq_length).expand(N, seq_length).to(self.device) # bz x seq
        # out = self.dropout(self.embed(x) + self.position_embed(position)) # word embed + positional embed
        out = self.dropout(x)
        for layer in self.layers:
            out = layer(out, out, out, mask) # encoder use same thing for query key value
        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        """
        src_mask: mask padding to avoid unnecessary computation
        trg_mask: mask the future position's token
        """
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x)) # (N, query_len = trg_len, hidden)
        out = self.transformer_block(value, key, query, src_mask)
        return out

class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_len,

    ):
        super(Decoder, self).__init__()
        self.device = device
        self.embed = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embed = nn.Embedding(max_len, embed_size)
    
        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
            for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.embed(x) + self.position_embed(positions)) # word embed + positional embed
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        out = self.fc_out(x)
        return out


class PatchEmbedding(nn.Module):
    def __init__(
        self, 
        in_channels = 3, 
        patch_size=28, 
        emb_size= 512, 
        img_size = 224,
        device='cpu'
    ):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2, emb_size))

        
    def forward(self, x):
        x = self.projection(x)
        # add position embedding
        x += self.positions
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        tokenizer,
        embed_size=512,
        num_layers=4,
        forward_expansion=4,
        heads=8,
        dropout=0.1,
        device="cpu",
        max_len=256,
    ):
        super(Transformer, self).__init__()
        self.tokenizer = tokenizer
        self.encoder = Encoder(
            tokenizer.vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_len
        )
        self.decoder = Decoder(
            tokenizer.vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_len
        )
        self.patch_image = PatchEmbedding()
        self.src_pad_idx = self.tokenizer.pad
        self.trg_pad_idx = self.tokenizer.pad
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2) # (N, 1, 1, src_len)
        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        # make source img into patches
        src, trg = src.to(self.device), trg.to(self.device)
        # img (N, channel=3, width=224, width) => (N, #patches, hidden)
        patches = self.patch_image(src) # (N, #patches, hidden)
        # src_mask = self.make_src_mask(patches) 
        # no padding in source so just a matrix of 1
        src_mask = torch.ones(patches.size(0), 1, 1, patches.size(1)).to(self.device)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(patches, src_mask) # (N, src_len, hidden)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
    
    def inference(self, img):
        pathches = self.patch_image(img)
        bz, src_len, embed_size = pathches.shape
        encoder_out = self.encoder(pathches, None) # no need for mask
        max_len = 100
        out = img.new_full((bz, max_len), self.tokenizer.pad).long()
        input_token = img.new_full((bz, 1), self.tokenizer.eos).long() # start with eos token
        is_decoding = img.new_full((bz, 1), 1).bool().squeeze(1)
        for i in range(max_len):
            decoder_out = self.decoder(input_token, encoder_out, None, None)
            logit = decoder_out[:, -1, :] 
            # padding has weight 0 so remember to diasable it
            logit[:, self.tokenizer.pad] = -float('inf')
            new_token = logit.argmax(dim=1).masked_fill_(
                ~is_decoding,
                self.tokenizer.pad # pad index
            )
            out[:, i] = new_token
            is_decoding = is_decoding * torch.ne(new_token, self.tokenizer.eos)
            if (torch.all(~is_decoding)):
                break
            # next step is predicted on all previous output unlike RNN that only needs prev 1
            input_token = torch.cat([input_token, new_token.unsqueeze(1)], dim=1)
        return out



if __name__ == '__main__':
    # test decoder given encoder output
    import sys, os
    sys.path.append(os.path.abspath(os.path.join('/mnt/d/Github/MICaptioning', '')))

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

    transformer = Transformer(tokenizer)
    # test regular MT transformer
    # src = torch.arange(0,10).view(2,5)
    # tgt = torch.arange(0,8).view(2,4)
    # out = transformer(src, tgt)

    for img, caption, tagvec in train_dataloader:
        out = transformer(img, caption)
        print(out.shape)
        break