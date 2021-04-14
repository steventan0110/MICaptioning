import torch

# train model for 1 epoch
class Trainer():
    def __init__(self, model, train_dataloader, valid_dataloader, **kwargs):
        self.args = kwargs
        self.device = torch.device('cpu' if self.args['cpu'] else 'cuda')
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        # TODO: constructing optimizer, scheduler can be modularized for different args passed in
        self.optimizer = optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=self.args.learning_rate, 
            epochs=self.args['max_epoch'], 
            steps_per_epoch=len(self.train_dataloader)) 

    def train(self):
        # train for 1 epoch and return the loss of both train and valid set
        train_loss = 0
        val_loss = 0
        train_steps = 0
        val_steps = 0
        self.model.train()
        for i, (img, caption) in enumerate(self.train_dataloader):
            img, caption = img.to(self.device), caption.to(self.device)
            loss = self.model(img, caption)
            train_loss += loss
            # backward, might want to control the update step size
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            train_steps += 1
        
        self.model.eval()
        with torch.no_grad():
            for i, (img, caption) in enumerate(self.valid_dataloader):
                img, caption = img.to(self.device), caption.to(self.device)
                loss = self.model(img, caption)
                valid_loss += loss
                val_steps += 1

        return train_loss / train_steps, val_loss / val_steps

    def save_checkpoint(self, EPOCH, PATH):
        torch.save({
            'epoch': EPOCH,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, PATH)

    def load_checkpoint(self, PATH):
        checkpoint = torch.load(PATH)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

