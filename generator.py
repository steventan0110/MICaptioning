import torch
import sacrebleu
import numpy as np

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
    
        scores = []
        for (img, caption) in self.test_dataloader:
            bz = img.size(0)
            tokens = self.model.inference(img)
            hypo = self.tokenizer.decode(tokens)
            tgt = self.tokenizer.decode(caption)
            print(hypo)
            print(tgt)
            # print()
            for i in range(bz):
                # compute bleu for each pair and print out if required
                bleu = sacrebleu.corpus_bleu(hypo[i], tgt[i])
                scores.append(bleu.score)

        score = np.array(scores).mean()
        print("Final BLEU Score averaged on all sentences: ", score)

        
    def beam_search(self, model, img):
        # TODO: perform beam search for the inferenced tokens
        pass
    
    def load_checkpoint(self, PATH):
        checkpoint = torch.load(PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
   
