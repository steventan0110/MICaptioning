import torch
import torch.nn as nn
import torchvision.models as models

# reference: https://github.com/ZexinYan/Medical-Report-Generation/blob/d26f25c628cb0d5626e7b70e8d49366a340003ac/utils/models.py

# CNN to produce visual features
class EncoderCNN(nn.Module):
	def __init__(self):
		super(EncoderCNN, self).__init__()

		vgg = False
		if vgg:
			cnn = models.vgg19(pretrained = False)
			#self.enc_dim = list(cnn.features.children())[-3].weight.shape[0]  # ?
		else:
			cnn = models.resnet152(pretrained = False)
		modules = list(cnn.children())[:-2]
		self.cnn = nn.Sequential(*modules)
		self.avgpool = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

	def forward(self, x):
		visual_features = self.cnn(x) # (batch_size, enc_dim, enc_img_size, enc_img_size)
		#visual_features = visual_features.permute(0, 2, 3, 1)
		avg_features = self.avgpool(visual_features).squeeze()
		return visual_features, avg_features

# MLC for predicting tags and use their embeddings as semantic features
class MLC(nn.Module):
    def __init__(self,
                 classes=156,
                 sementic_features_dim=512,
                 fc_in_features=2048,
                 k=10):
        super(MLC, self).__init__()
        self.classifier = nn.Linear(in_features=fc_in_features, out_features=classes)
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
        self.W_v_h = nn.Linear(in_features=hidden_size, out_features=visual_size)
        self.W_v_att = nn.Linear(in_features=visual_size, out_features=visual_size)

        self.W_a = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.W_a_h = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.W_a_att = nn.Linear(in_features=hidden_size, out_features=hidden_size)

        self.W_fc = nn.Linear(in_features=visual_size + hidden_size, out_features=embed_size)

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
	encoder = EncoderCNN()
	print(encoder)