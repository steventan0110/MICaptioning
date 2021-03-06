import torch
import sacrebleu
import numpy as np
from utils.search import SingleBeamSearchBoard

class Generator():
    def __init__(self, model, model_path, test_dataloader, tokenizer, **kwargs):
        self.args = kwargs
        self.device = torch.device('cpu' if self.args['cpu'] else 'cuda')
        self.beam_size = self.args['beam_size']
        self.model = model.to(self.device)
        self.load_checkpoint(model_path)
        self.tokenizer = tokenizer
        self.test_dataloader = test_dataloader

    def eval(self):
        # TODO: generate result and compute BLEU score
        tgt = []
        hypo = []
        scores = []
        for (img, caption, tags_vec) in self.test_dataloader:
            bz = img.size(0)
            tokens = self.model.inference(img)
            # tokens, _ = self.model.beam_search(self.beam_size, self.device, SingleBeamSearchBoard, img)
            # tokens = torch.tensor(tokens)
            hypo = self.tokenizer.decode(tokens)
            tgt = self.tokenizer.decode(caption)
            # print('tgt:', tgt)
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
   
