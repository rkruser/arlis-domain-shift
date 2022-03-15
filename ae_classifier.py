from ae_combined import *
from utils import EasyDict as edict



class Classifier(nn.Module):
    def __init__(self, input_size = 32, num_out=1, use_features=True, lr=0.0001):
        super().__init__()
        self.encoder = Encoder(input_size, add_linear=True, use_layer_norm=True)
        if use_features:
            self.feature_encoder = Feature_Encoder(linear=False)
        else:
            self.feature_encoder = None
        self.final_output = nn.Linear(512, 1)

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x, features=None):
        x = self.encoder(x)
        if self.feature_encoder is not None:
            x = self.feature_encoder(x, features)
        x = self.final_output(x).squeeze(1)
        return x

    def step_optim(self):
        self.optim.step()



def train(model, dataloader, n_epochs=100, use_features=True, print_every=100):
    print(model)
    model.cuda()
    model.train()

    bce_lossfunc = torch.nn.BCEWithLogitsLoss()

    for epoch in range(n_epochs):
        print("***************************epoch",epoch, "*************************")
        for i, batch in enumerate(dataloader):
            x_real = batch['real'][0].cuda()
            x_fake = batch['fake'][1].cuda()
            x_aug = batch['augmented'][0].cuda()

            if use_features:
                x_real_feats = batch['real'][-2].cuda()
                x_fake_feats = batch['fake'][-2].cuda()
                x_aug_feats = batch['augmented'][-2].cuda()
            else:
                x_real_feats = None
                x_fake_feats = None
                x_aug_feats = None

            real_pred = model(x_real, features=x_real_feats)
            fake_pred = model(x_fake, features=x_fake_feats)
            aug_pred = model(x_aug, features=x_aug_feats)
            
            real_labels = torch.ones_like(real_pred)
            synth_labels = torch.ones_like(fake_pred)
            aug_labels = torch.zeros_like(aug_pred)

            real_loss = bce_lossfunc(real_pred, real_labels)
            synth_loss = bce_lossfunc(fake_pred, synth_labels)
            aug_loss = bce_lossfunc(aug_pred, aug_labels)

            total_loss = real_loss + synth_loss + aug_loss


            if (i%print_every == 0):
                loss_str = "real: {0}, synth: {1}, aug: {2}".format(real_loss.item(), synth_loss.item(), aug_loss.item())
                print(i)
                print(loss_str)


            model.zero_grad()
            total_loss.backward()
            model.step_optim()








































































