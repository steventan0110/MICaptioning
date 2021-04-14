import torch
import sacrebleu

class Generator():
    def __init__(self, model, model_path, test_dataloader, tokenizer):
        self.args = kwargs
        self.device = torch.device('cpu' if self.args['cpu'] else 'cuda')
        self.model = model.to(self.device)
        self.load_checkpoint(model_path)
        self.tokenizer = tokenizer

    def eval():
        # TODO: generate result and compute BLEU score
        tgt = []
        hypo = []
        for (img, caption) in test_dataloader:
            caption_text = self.tokenizer.decode(caption)
            tgt.append(caption_text)
            img = img.to(self.device)
            decoder_out = self.beam_search(model, img)
            hypo.append(decoder_out)
        
        bleu = sacrebleu.corpus_bleu(tgt, [hypo])
        print(bleu.score)
        
    def beam_search(self, model, img):
        # TODO: perform beam search for the inferenced tokens
        pass
    
    def load_checkpoint(self, PATH):
        checkpoint = torch.load(PATH)
        self.model.load_state_dict(checkpoint['model_state_dict'])
   
