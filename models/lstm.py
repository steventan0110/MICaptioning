import torch
from decoder import Decoder

class LSTMDecodeer(Decoder):
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1,
    ):
        super().__init__(dictionary)
        # model params defined here


    def forward(self):
        pass

    def extract_features(self, prev_output, encoder_out):
        pass