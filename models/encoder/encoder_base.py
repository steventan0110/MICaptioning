import torch.nn as nn

class Encoder(nn.Module):
    
    def __init__(self, dictionary):
        super().__init__() 
        self.dict = dictionary
    

    def forward(self, prev_output, encoder_out=None, **kwargs):
        # compute the output and associated states
        x, extra = self.extract_features(prev_output, encoder_out=encoder_out, **kwargs)
        return self.output_layer(x), extra
    
    def extract_features(self, prev_output, encoder_out=None, **kwargs):
        raise NotImplementedError

    def output_layer(self, input, **kwargs):
        raise NotImplementedError

    