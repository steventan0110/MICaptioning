from pathlib import Path
import argparse
from utils.tokenizer import Tokenizer
from utils.dataset import ChestXrayDataSet, collate_fn
from torchvision import transforms
import torch
from trainer import Trainer
from generator import Generator
from models.model_base import EncoderDecoderModel
from models.transformer import Transformer

def main(args):
    print(args, flush=True)
    tokenizer = Tokenizer(args.caption_dir)
    transform = transforms.Compose([
        transforms.ToTensor(),
        # resize is necessary!
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225))
    ])

    if args.mode == 'train': 
        train_dataset = ChestXrayDataSet(args.data_dir, 'train', tokenizer, transform)
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              collate_fn=collate_fn)

        valid_dataset = ChestXrayDataSet(args.data_dir, 'valid', tokenizer, transform=valid_transform)
        valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              collate_fn=collate_fn)

        if args.arch == 'transformer':
            model = Transformer(tokenizer, device=torch.device('cpu' if args.cpu else 'cuda'))
        else:
            model = EncoderDecoderModel(args.arch.split('_')[0], tokenizer)

        trainer = Trainer(model, train_dataloader, valid_dataloader, **vars(args))
        if args.load_dir:
            trainer.load_checkpoint(args.load_dir)

        EPOCH = args.max_epoch 
        for i in range(EPOCH):
            # trainer perform forward calculation for only 1 epoch
            loss, val_loss = trainer.train()
            print('Epoch{}, loss:{}, validation loss{}'.format(i+1, loss, val_loss), flush=True)
            if i % args.log_interval == 0:
                trainer.save_checkpoint(i, loss, val_loss, args.save_dir)

    elif args.mode == 'test':
        # TODO: decode image and compute BLEU against test captions
        test_dataset = ChestXrayDataSet(args.data_dir, 'test', tokenizer, transform=valid_transform)
        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              collate_fn=collate_fn)

        if args.arch == 'transformer':
            model = Transformer(tokenizer, device=torch.device('cpu' if args.cpu else 'cuda'))
        else:
            model = EncoderDecoderModel(args.arch.split('_')[0], tokenizer)
        model.eval()
        generator = Generator(model, args.load_dir, test_dataloader, tokenizer, **vars(args))
        generator.eval() # print to console the evaluation

    elif args.mode == 'interactive':
        test_dataset = ChestXrayDataSet(args.data_dir, 'test', tokenizer, transform=valid_transform)
        # idx = input("input test image id:")
        # x = test_dataset[idx]
        # img, caption, tags_vec = collate_fn([test_dataset[idx]])
        test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      collate_fn=collate_fn)
        for img, caption, tags_vec in test_dataloader:
            if not args.cpu:
                img = img.cuda()
                caption = caption.cuda()

        model = EncoderDecoderModel(args.arch.split('_')[0], tokenizer)
        if not args.cpu:
            model = model.cuda()
        if args.load_dir:
            checkpoint = torch.load(args.load_dir)
            model.load_state_dict(checkpoint['model_state_dict'])

        tokens = model.inference(img)
        hypo = model.tokenizer.decode(tokens)
        tgt = model.tokenizer.decode(caption)
        print('predicted:', hypo)
        print('target:', tgt)
        temp = tags_vec.long().numpy().tolist()[0]
        indices = [i for i in range(len(temp)) if temp[i] > 0]
        tags = model.tag_transform.array2tags(indices)
        tags_pred = model.tag_inference(img, 4)
        print('predicted tag:', tags_pred)
        print('ground truth tag:', tags)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--mode', '-m', choices={'train', 'test', 'interactive'}, help="execution mode")
    parser.add_argument('--caption-dir', type=Path)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--save-dir', type=Path)
    parser.add_argument('--load-dir', default=None, type=Path)
    parser.add_argument('--cpu', default=False, action="store_true")
    parser.add_argument('--arch', default="vgg_lstm")
    parser.add_argument('--max-epoch', default=100, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--learning-rate', '-lr', default=0.001, type=float)
    parser.add_argument('--log-interval', default=1, type=int)
    parser.add_argument('--beam-size', default=5, type=int)
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
