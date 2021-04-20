import torch.nn as nn

class Decoder(nn.Module):
    
    def __init__(self, dictionary):
        super().__init__() 
        self.dict = dictionary
    

    def forward(self, prev_output, encoder_out=None, **kwargs):
        # compute the output and associated states
        raise NotImplementedError


    