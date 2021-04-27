import torch
import torch.nn as nn
import torchvision.models as models

# reference: https://github.com/ZexinYan/Medical-Report-Generation/blob/d26f25c628cb0d5626e7b70e8d49366a340003ac/utils/models.py

# CNN to produce visual features
class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        vgg = True
        if vgg:
            cnn = models.vgg19(pretrained=True)
            # self.enc_dim = list(cnn.features.children())[-3].weight.shape[0]  # ?
        else:
            cnn = models.resnet152(pretrained=True)
        modules = list(cnn.children())[:-2]
        self.cnn = nn.Sequential(*modules)
        self.avgpool = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, x):
        # (batch_size, enc_dim, enc_img_size, enc_img_size)
        visual_features = self.cnn(x) # batch_size x 512 channels x 7 x 7
        # print('visual feature shape:', visual_features.shape)

        avg_features = self.avgpool(visual_features).squeeze() # batch_size x 512 channels
        # print(('avg feature shape:', avg_features.shape))
        return visual_features, avg_features

# MLC for predicting tags and use their embeddings as semantic features
class MLC(nn.Module):
    def __init__(self,
                 classes=156,
                 sementic_features_dim=512,
                 fc_in_features=2048,
                 k=10):
        super(MLC, self).__init__()
        self.classifier = nn.Linear(
            in_features=fc_in_features, out_features=classes)
        self.embed = nn.Embedding(classes, sementic_features_dim)
        self.softmax = nn.Softmax()
        self.__init_weight()

    def __init_weight(self):
        self.classifier.weight.data.uniform_(-0.1, 0.1)
        self.classifier.bias.data.fill_(0)

    def forward(self, avg_features):
        tags = self.softmax(self.classifier(avg_features))
        semantic_features = self.embed(torch.topk(tags, self.k)[1])
        return tags, semantic_features


class CoAttention(nn.Module):
    def __init__(self,
                 embed_size=512,
                 hidden_size=512,
                 visual_size=2048,
                 k=10):
        super(CoAttention, self).__init__()
        self.W_v = nn.Linear(in_features=visual_size, out_features=visual_size)
        self.W_v_h = nn.Linear(in_features=hidden_size,
                               out_features=visual_size)
        self.W_v_att = nn.Linear(
            in_features=visual_size, out_features=visual_size)

        self.W_a = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.W_a_h = nn.Linear(in_features=hidden_size,
                               out_features=hidden_size)
        self.W_a_att = nn.Linear(
            in_features=hidden_size, out_features=hidden_size)

        self.W_fc = nn.Linear(in_features=visual_size +
                              hidden_size, out_features=embed_size)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, avg_features, semantic_features, h_sent):
        W_v = self.W_v(avg_features)
        W_v_h = self.W_v_h(h_sent.squeeze(1))
        alpha_v = self.softmax(self.W_v_att(self.tanh(W_v + W_v_h)))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.W_a_h(h_sent)
        W_a = self.W_a(semantic_features)
        alpha_a = self.softmax(self.W_a_att(self.tanh(torch.add(W_a_h, W_a))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a


if __name__ == '__main__':
    # test encoder
    import sys, os
    sys.path.append(os.path.abspath(os.path.join('../..', '')))
    sys.path.append('/Users/chenyuzhang/Desktop/JHU-6/DL/MICaptioning')

    from utils.tokenizer import Tokenizer
    from utils.dataset import ChestXrayDataSet, collate_fn
    from torchvision import transforms

    #caption_dir = '/mnt/d/Github/MICaptioning/iu_xray/iu_xray_captions.json'
    #data_dir = '/mnt/d/Github/MICaptioning/datasets'
    caption_dir = '/Users/chenyuzhang/Desktop/JHU-6/DL/MICaptioning/iu_xray/iu_xray_captions.json'
    data_dir = '/Users/chenyuzhang/Desktop/JHU-6/DL/MICaptioning/datasets'
    tokenizer = Tokenizer(caption_dir)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    train_dataset = ChestXrayDataSet(data_dir, 'train', tokenizer, transform)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=4,
                                                   shuffle=False,
                                                   collate_fn=collate_fn)
    encoder = EncoderCNN()
    print(encoder)
    for img, caption in train_dataloader:
        # test encoding img into features
        print(img.shape)
        encoder_out = encoder(img)
        break

