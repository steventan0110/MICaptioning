import torch
import sacrebleu

class Generator():
    def __init__(self, model, model_path, test_dataloader, tokenizer, **kwargs):
        self.args = kwargs
        self.device = torch.device('cpu' if self.args['cpu'] else 'cuda')
        self.model = model.to(self.device)
        self.load_checkpoint(model_path)
        self.tokenizer = tokenizer
        self.test_dataloader = test_dataloader
     
    def eval(self):
        # TODO: generate result and compute BLEU score
        tgt = []
        hypo = []
        # cell and LSTM has to have the same parameter!
        # print(list(self.model.LSTMCell.named_parameters()))
        # print(list(self.model.decoder.lstm.named_parameters()))
        
        for (img, caption) in self.test_dataloader:
            tokens = self.model.inference(img, caption)
            # TODO: decode and compute score
        
        # bleu = sacrebleu.corpus_bleu(tgt, [hypo])
        # print(bleu.score)
        
    def beam_search(self, model, img):
        # TODO: perform beam search for the inferenced tokens
        pass
    
    def load_checkpoint(self, PATH):
        checkpoint = torch.load(PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
   
