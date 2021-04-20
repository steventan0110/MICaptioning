from pathlib import Path
import argparse
from utils.tokenizer import Tokenizer
from utils.dataset import ChestXrayDataSet, collate_fn
from torchvision import transforms
import torch
from trainer import Trainer
from generator import Generator

def main(args):
    print(args)
    tokenizer = Tokenizer(args.caption_dir)
    transform = transforms.Compose([
        transforms.ToTensor(),
        # resize is necessary!
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
                        
    if args.mode == 'train': 
        train_dataset = ChestXrayDataSet(args.data_dir, 'train', tokenizer, transform)
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              collate_fn=collate_fn)

        valid_dataset = ChestXrayDataSet(args.data_dir, 'valid', tokenizer, transform=None)
        valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              collate_fn=collate_fn)

        # TODO: load model here once model file is written
        model = None
        trainer = Trainer(model, train_dataloader, valid_dataloader, args)
        if args.load_dir:
            trainer.load_checkpoint(args.load_dir)

        EPOCH = args.max_epoch 
        for i in range(EPOCH):
            # trainer perform forward calculation for only 1 epoch
            loss, val_loss = trainer.train()
            print('Epoch{}, loss:{}, validation loss{}'.format(i+1, loss, val_loss))
            if i % args.log_interval == 0:
                trainer.save_checkpoint(i, args.save_dir)

    else:
        # TODO: decode image and compute BLEU against test captions
        test_dataset = ChestXrayDataSet(args.data_dir, 'test', tokenizer, transform=None)
        test_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              collate_fn=collate_fn)
        
        # TODO: load model here once model file is written
        model = None
        generator = Generator(model, args.load_dir, test_dataloader, tokenizer)
        generator.eval() # print to console the evaluation

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--mode', '-m', choices={'train', 'test'}, help="execution mode")
    parser.add_argument('--caption-dir', type=Path)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--save-dir', type=Path)
    parser.add_argument('--load-dir', default=None, type=Path)
    parser.add_argument('--cpu', default=False, action="store_true")
    parser.add_argument('--arch', default="VGGLSTM")
    parser.add_argument('--max-epoch', default=100, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--learning-rate', '-lr', default=0.001, type=float)
    parser.add_argument('--log-interval', default=1, type=int)
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
