import torch
import os
# train model for 1 epoch
class Trainer():
    def __init__(self, model, train_dataloader, valid_dataloader, **kwargs):
        self.args = kwargs
        self.device = torch.device('cpu' if self.args['cpu'] else 'cuda')
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        # TODO: constructing optimizer, scheduler can be modularized for different args passed in
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=self.args['learning_rate'], 
            epochs=self.args['max_epoch'], 
            steps_per_epoch=len(self.train_dataloader)) 

    def train(self):
        # train for 1 epoch and return the loss of both train and valid set
        train_loss = 0
        val_loss = 0
        train_steps = 0
        val_steps = 0
        self.model.train()
        for i, (img, caption, tags_vec) in enumerate(self.train_dataloader):
            img, caption = img.to(self.device), caption.to(self.device)
            out = self.model(img, caption)
            # need to flatten the sentence before computing crossentropy loss
            vocab_size = out.size(2)
            loss = self.criterion(out.reshape(-1, vocab_size), caption.reshape(-1, 1).squeeze())
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
                img, caption = img.to(self.device), caption.to(self.device)
                out = self.model(img, caption)
                vocab_size = out.size(2)
                loss = self.criterion(out.reshape(-1, vocab_size), caption.reshape(-1, 1).squeeze())
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

