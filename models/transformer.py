import torch
import torch.nn as nn

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

        if mask is not None:
            energy = energy.masked_fill(mask==0, float("-1e20"))
        
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
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size), # some extra mapping to higher dimension
            nn.ReLu(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        # skip connect
        x = self.dropout(self.norm1(attention+query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x)) # skip connect


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        emebd_size=512,
        num_layers,
        heads,
        device,
        forward_expansion=4,
        dropout=0.2,
        max_len, # limit for positional embedding to work
    ):
        super(Encoder, self).__init__()
        self.embed_size = emebd_size
        self.device = device
        self.embed = nn.Embedding(src_vocab_size, emebd_size)
        self.position_embed = nn.Embedding(max_len, emebd_size)
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
        N, seq_length = x.shape
        position = torch.arange(0, seq_length).expand(N, seq_length).to(self.device) # bz x seq
        out = self.dropout(self.embed(x) + self.position_embed(position)) # word embed + positional embed

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
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out

class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        emebd_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_len,

    ):
        super(Decoder, self).__init__()
        self.device = device
        self.embed = nn.Embedding(trg_vocab_size, emebd_size)
        self.position_embed = nn.Embedding(max_len, emebd_size)
    
        self.layers = nn.ModuleList(
            [DecoderBlock(emebd_size, heads, forward_expansion, dropout, device)
            for _ in range(number_layers)]
        )
        self.fc_out = nn.Linear(emebd_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length),to(self.device)
        x = self.dropout(self.embed(x) + self.position_embed(position)) # word embed + positional embed
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        out = self.fc_out
        return out

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        tgt_pad_idx,
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cuda",
        max_len=100,
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_len
        )
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_len
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
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
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
        



