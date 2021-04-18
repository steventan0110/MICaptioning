import torch
from decoder import Decoder

class LSTMDecoder(Decoder):
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1,
    ):
        super().__init__(dictionary)
        # model params defined here
        vocab_size = #
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, prev_state):
        embeddings = self.embed(x)
        output, state = self.lstm(embeddings, prev_state)
        logits = self.linear(output)
        return logits, state

    def extract_features(self, prev_output, encoder_out):
        pass