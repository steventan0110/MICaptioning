import torch
import os
import torch.nn.functional as F
# train model for 1 epoch

class Trainer():
    def __init__(self, model, train_dataloader, valid_dataloader, tl_scale=1, **kwargs):
        self.args = kwargs
        self.device = torch.device('cpu' if self.args['cpu'] else 'cuda')
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.tl_scale = tl_scale
        # TODO: constructing optimizer, scheduler can be modularized for different args passed in
        self.criterion = torch.nn.CrossEntropyLoss()
        self.kldiv = torch.nn.KLDivLoss()
        self.optimizer = optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=self.args['learning_rate'], 
            epochs=self.args['max_epoch'], 
            steps_per_epoch=len(self.train_dataloader)) 

    def right_shift(self, tokens):
        eos = self.train_dataloader.dataset.tokenizer.eos
        pad = self.train_dataloader.dataset.tokenizer.pad
        eos_indice = (tokens == eos).nonzero(as_tuple=True)[1]
        bz, src_len = tokens.shape
        temp = tokens.roll(1, 1)
        temp[:, 0] = eos
        for i in range(bz):
            if eos_indice[i] < src_len - 1:
                temp[i, eos_indice[i]+1] = pad
        return temp

    def train(self):
        # train for 1 epoch and return the loss of both train and valid set
        train_loss = 0
        val_loss = 0
        train_steps = 0
        val_steps = 0
        self.model.train()
        for i, (img, caption, tags_vec) in enumerate(self.train_dataloader):
            img, caption, tags_vec = img.to(self.device), caption.to(self.device), tags_vec.to(self.device)
            tags_distrib = F.normalize(tags_vec, dim=1)
            # need to right shift for transformer architecture
            if self.args['arch'] == 'transformer':
                prev_caption = self.right_shift(caption)
            else:
                prev_caption = caption
            
            out, tags_pred = self.model(img, prev_caption)
            # need to flatten the sentence before computing crossentropy loss
            vocab_size = out.size(2)
            cap_loss = self.criterion(out.reshape(-1, vocab_size), caption.reshape(-1, 1).squeeze())
            tag_loss = self.kldiv(tags_pred, tags_distrib)
            loss = cap_loss + self.tl_scale * tag_loss
            train_loss += loss
            # backward, might want to control the update step size
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            train_steps += 1

        self.model.eval()
        with torch.no_grad():
            for i, (img, caption, tags_vec) in enumerate(self.valid_dataloader):
                img, caption, tags_vec = img.to(self.device), caption.to(self.device), tags_vec.to(self.device)
                tags_distrib = F.normalize(tags_vec, dim=1)
                if self.args['arch'] == 'transformer':
                    prev_caption = self.right_shift(caption)
                else:
                    prev_caption = caption
                
                out, tags_pred = self.model(img, prev_caption)
                vocab_size = out.size(2)
                cap_loss = self.criterion(out.reshape(-1, vocab_size), caption.reshape(-1, 1).squeeze())
                tag_loss = self.kldiv(tags_pred, tags_distrib)
                loss = cap_loss + self.tl_scale * tag_loss
                val_loss += loss
                val_steps += 1

        return train_loss / train_steps, val_loss / val_steps

    def save_checkpoint(self, EPOCH, loss, val_loss, PATH):

        save_path = os.path.join(PATH, 'checkpoint'+str(EPOCH)+'.pt')
        torch.save({
            'epoch': EPOCH,
            'loss': loss,
            'val_loss': val_loss,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, save_path)

    def load_checkpoint(self, PATH):
        checkpoint = torch.load(PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

