import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, in_feats, act=nn.Mish, pretrained_model=None):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.embedding_layer = nn.Sequential(
            nn.Linear(in_feats, 128)
        )
        self.encoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.Dropout(p=0),
            act(),
            nn.Linear(64, 32),
            nn.Dropout(p=0),
            act(),
            nn.Linear(32, 128)
        )
        self.predictor = nn.Sequential(
            nn.Linear(128, 32),
            nn.Dropout(p=0),
            act(),   
            nn.Linear(32, 1),
        )

    def forward(self, cat_feat, num_feat=None):
        if self.pretrained_model is not None:
            pretrained_embed = self.pretrained_model(cat_feat, num_feat, get_embedding=True)
            embedding = self.embedding_layer(pretrained_embed)
            encoder_output = self.encoder(embedding)
            pred = self.predictor(encoder_output)
            return pred, embedding
        else:
            embedding = self.embedding_layer(cat_feat)
            pred = self.predictor(embedding)
            return pred, embedding
        

class Encoder(nn.Module):
    def __init__(self, categorical, numerical, out_dim, act=nn.Mish):
        super().__init__()
        categorical_dim = 10
        self.embed0 = nn.Embedding(categorical[0], categorical_dim)
        self.embed1 = nn.Embedding(categorical[1], categorical_dim)
        self.embed2 = nn.Embedding(categorical[2], categorical_dim)
        self.embed3 = nn.Embedding(categorical[3], categorical_dim)
        self.embed4 = nn.Embedding(categorical[4], categorical_dim)
        self.embed5 = nn.Embedding(categorical[5], categorical_dim)
        self.embed6 = nn.Embedding(categorical[6], categorical_dim)
        self.embed7 = nn.Embedding(categorical[7], categorical_dim)
        self.embed8 = nn.Embedding(categorical[8], categorical_dim)
        self.embed9 = nn.Embedding(categorical[9], categorical_dim)
        self.embed10 = nn.Embedding(categorical[10], categorical_dim)
        self.embed11 = nn.Embedding(categorical[11], categorical_dim)
        self.embed12 = nn.Embedding(categorical[12], categorical_dim)
        self.embed13 = nn.Embedding(categorical[13], categorical_dim)
        self.embed14 = nn.Embedding(categorical[14], categorical_dim)
        self.embed15 = nn.Embedding(categorical[15], categorical_dim)
        if len(categorical) == 17:
            self.embed16 = nn.Embedding(categorical[16], categorical_dim)

        self.num_embed = nn.Sequential(
            nn.Linear(numerical, 64),
            nn.Dropout(p=0.1),
            act(),
            nn.Linear(64, 32)
        )

        self.encoder = nn.Sequential(
            nn.Linear(len(categorical)*categorical_dim+32, 256),
            nn.Dropout(p=0.1),
            act(),
            nn.Linear(256, 512),
            nn.Dropout(p=0.1),
            act(),            
            nn.Linear(512, 1024),
            nn.Dropout(p=0.1),
            act(),
            nn.Linear(1024, 512),
            nn.Dropout(p=0.1),
            act(),
            nn.Linear(512, 256)
        )
        self.pretrained_layer = nn.Sequential(
            nn.Linear(256, out_dim)
        )

    def forward(self, categorical, numerical, get_embedding = False):
        output0 = self.embed0(categorical[:, 0])
        output1 = self.embed1(categorical[:, 1])
        output2 = self.embed2(categorical[:, 2])
        output3 = self.embed3(categorical[:, 3])
        output4 = self.embed4(categorical[:, 4])
        output5 = self.embed5(categorical[:, 5])
        output6 = self.embed6(categorical[:, 6])
        output7 = self.embed7(categorical[:, 7])
        output8 = self.embed8(categorical[:, 8])
        output9 = self.embed9(categorical[:, 9])
        output10 = self.embed10(categorical[:, 10])
        output11 = self.embed11(categorical[:, 11])
        output12 = self.embed12(categorical[:, 12])
        output13 = self.embed13(categorical[:, 13])
        output14 = self.embed14(categorical[:, 14])
        output15 = self.embed15(categorical[:, 15])
        if len(categorical[0]) == 17:
            output16 = self.embed16(categorical[:, 16])

        total = output0
        total = torch.cat((total, output1), 1)
        total = torch.cat((total, output2), 1)
        total = torch.cat((total, output3), 1)
        total = torch.cat((total, output4), 1)
        total = torch.cat((total, output5), 1)
        total = torch.cat((total, output6), 1)
        total = torch.cat((total, output7), 1)
        total = torch.cat((total, output8), 1)
        total = torch.cat((total, output9), 1)
        total = torch.cat((total, output10), 1)
        total = torch.cat((total, output11), 1)
        total = torch.cat((total, output12), 1)
        total = torch.cat((total, output13), 1)
        total = torch.cat((total, output14), 1)
        total = torch.cat((total, output15), 1)
        if len(categorical[0]) == 17:
            total = torch.cat((total, output16), 1)
        
        imp_num = self.num_embed(numerical)
        total = torch.cat((total, imp_num), 1)

        if get_embedding:
            return self.encoder(total)
        else:
            embedding = self.encoder(total)
            output = self.pretrained_layer(embedding)
            return output, embedding

