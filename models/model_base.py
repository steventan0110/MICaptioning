import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder import Decoder
from encoder import Encoder
class EncoderDecoderModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        assert isinstance(self.encoder, Encoder)
        assert isinstance(self.decoder, Decoder)

    def forward(self, src_images, **kwargs):
        encoder_out = self.encoder(src_images, **kwargs)
        decoder_out = self.decoder(prev_output, encoder_out=encoder_out, **kwargs)
        return decoder_out

    
        
