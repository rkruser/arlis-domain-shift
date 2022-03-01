from ae_models import *
from utils import EasyDict as edict





class Feature_Encoder(nn.Module):
    def __init__(self, linear=False, nfeats=1000):
        super().__init__()

        if linear:
            self.nonlinearity = nn.Identity()
        else:
            self.nonlinearity = nn.LeakyReLU(0.2,inplace=True)

        self.encode = nn.Linear(512 + nfeats, 512)

#        self.block = nn.Sequential(
#            self.nonlinearity,
#            nn.Linear(512,512),
#            self.nonlinearity,
#            nn.Linear(512,512),
#            self.nonlinearity,
#            nn.Linear(512,512),
#            self.nonlinearity
#            )

        self.final = nn.Linear(512,512)


    def forward(self, x, features):
        x = torch.cat([x,features], dim=1)
        x = self.encode(x)
        x = self.nonlinearity(x)
        x = self.final(x)

        return x
 





class Feature_Decoder(nn.Module):
    def __init__(self, linear=False, nfeats=1000):
        super().__init__()

        if linear:
            self.nonlinearity = nn.Identity()
        else:
            self.nonlinearity = nn.LeakyReLU(0.2,inplace=True)

        self.initial = nn.Linear(512,512)
        self.nfeats = nfeats

#        self.block = nn.Sequential(
#            nn.Linear(512,512),
#            self.nonlinearity,
#            nn.Linear(512,512),
#            self.nonlinearity,
#            nn.Linear(512,512)
#            )

        self.decode = nn.Linear(512,512+nfeats)

    def forward(self, x):
        x = self.initial(x)
        x = self.nonlinearity(x)
        x = self.decode(x)
        x, features = torch.split(x, [512,self.nfeats], dim=1)

        return x, features



# Need to make dimensions for feature encoder / decoder line up with 513
# Phi after feature encoder, not before?

class Combined_Autoencoder:
    def __init__(self, global_lr=0.001, use_features=True, device='cuda:0'):

        self.modules = edict()
        self.modules.encoder = Encoder(32, ncolors=3, lean=False, very_lean=False, all_linear=False, add_linear=False).to(device)
        self.modules.decoder = Decoder(32, ncolors=3, lean=False, very_lean=False, all_linear=False, add_linear=False).to(device)
        self.modules.e2z = Phi(num_out=513).to(device) # 513 due to separate norm and direction
        self.modules.z2e = Phi(num_in=513).to(device)
        self.modules.adversary = Small_Classifier(linear=False, in_feats=513).to(device)

        
        if use_features:
            self.modules.feature_encode = Feature_Encoder().to(device)
            self.modules.feature_decode = Feature_Decoder().to(device)
        self.use_features = use_features


        self.optimizers = edict()
        for network in self.modules:
            self.optimizers[network] = torch.optim.Adam(self.modules[network].parameters(), lr=global_lr)

           


    def train(self):
        for key in self.modules:
            self.modules[key].train()

    def eval(self):
        for key in self.modules:
            self.modules[key].eval()


    # Think about how to do the normalization
    def encode(self, x, features=None):
        e = self.modules.encoder(x)
        e = e.reshape(e.size(0),-1)

        if self.use_features:
            e = self.modules.feature_encode(e,features)

        z = self.modules.e2z(e)

        z_norms, z_normalized = torch.split(z, [1, 512], dim=1)
        z_norms = torch.nn.functional.elu(z_norms) + 1 #make everything positive
        z_normalized = z_normalized / torch.norm(z_normalized, dim=1).unsqueeze(1)


#        z_norms = torch.norm(z,dim=1)
#        z_normalized = z / z_norms
        return z_norms.squeeze(1), z_normalized
        

    def decode(self, z_norms, z_normalized):
        z_concat = torch.cat([z_norms.unsqueeze(1), z_normalized],dim=1)
        e = self.modules.z2e(z_concat)

        if self.use_features:
            e, features = self.modules.feature_decode(e)
        else:
            features = None
        
        e = e.reshape(e.size(0), 32, 4, 4)
        x = self.modules.decoder(e)
        return x, features
        

    def adversary(self, z_norms, z_normalized):
        z_concat = torch.cat([z_norms.unsqueeze(1), z_normalized],dim=1)
        predicted = self.modules.adversary(z_concat)
        return predicted
        

    def zero_grad(self, mode='regular'):
        if mode == 'regular':
            for key in self.modules:
                if key != 'adversary':
                    self.modules[key].zero_grad() 
        elif mode=='adversary':
            self.modules.adversary.zero_grad()
        else:
            raise ValueError("Invalid zero_grad mode")



    def step_optim(self, mode='regular'):
        if mode == 'regular':
            for key in self.optimizers:
                if key != 'adversary':
                    self.optimizers[key].step()
        elif mode=='adversary':
            self.optimizers.adversary.step()
        else:
            raise ValueError("Invalid step mode")
       
# train stages:
#  - Train with fake/real data and adversary loss until z values have the right magnitude
#  - Then add augmented data and pressure loss
#  - Get jacobian probabilities of the phi networks with respect to unnormalized z (or normalized z if doing e2z?)
        

# x is a 1d torch tensor of vector norms
# Gradients of total loss are (approximately) -1 at T + S/k and 1 at T-S/k
# Gradients at T+S are zero at (approximately) T+S
def ring_lossfunc(x, dim=512, significance=3, sigma=0.7, k=10, eps=1e-4):
    T = dim**(0.5)
    S = significance*sigma
    A = 0.5*(S/k)**3
    B = 1/(2*(k**3)*(T+S))

    ring_loss = A/((x-T).square() + eps)
    reg_loss = B*x.square()
    total_loss = (ring_loss+reg_loss).mean()
    return total_loss

def train_combined(model, dataloader, n_epochs, use_features=False, ring_loss_after=10, ring_loss_max=10000,
                   lmbda_norm = 1, lmbda_cosine=1, lmbda_recon=1, lmbda_feat=1, lmbda_adv=1,
                   lmbda_ring=1
                   ):

    model.train()

    l1_lossfunc = torch.nn.MSELoss()
    l2_lossfunc = torch.nn.L1Loss()
    bce_lossfunc = torch.nn.BCEWithLogitsLoss()

    def recon_lossfunc(x,y):
        return l1_lossfunc(x,y)+l2_lossfunc(x,y)

    phase = 0
    for epoch in range(n_epochs):
        print("*******epoch",epoch)
        for i, batch in enumerate(dataloader):
            x_real = batch['real'][0].cuda()
            x_fake = batch['fake'][1].cuda()
            z_fake = batch['fake'][0].cuda()
            z_fake_norms = torch.norm(z_fake,dim=1).detach()
            z_fake_sphere = (z_fake/z_fake_norms.unsqueeze(1)).detach()
            x_aug = batch['augmented'][0].cuda()

            if use_features:
                x_real_feats = batch['real'][-2].cuda()
                x_fake_feats = batch['fake'][-2].cuda()
                x_aug_feats = batch['augmented'][-2].cuda()
            else:
                x_real_feats = None
                x_fake_feats = None
                x_aug_feats = None

            loss_info = None
            if phase == 0: #encoder/decoder
                z_real_pred_norms, z_real_pred_sphere = model.encode(x_real, features=x_real_feats)
                z_fake_pred_norms, z_fake_pred_sphere = model.encode(x_fake, features=x_fake_feats)
                z_aug_pred_norms, z_aug_pred_sphere = model.encode(x_aug, features=x_aug_feats)
                
                # Z losses and adversary losses
                z_norm_loss = (z_fake_pred_norms - z_fake_norms).square().mean()
                z_cosine_loss = 1 - (z_fake_sphere*z_fake_pred_sphere).sum(dim=1).mean()
                # use l2 instead of cosine? or does it not matter?

                z_real_label_pred = model.adversary(z_real_pred_norms, z_real_pred_sphere)
                z_real_label = torch.ones(z_real_label_pred.size(),device=z_real_label_pred.device)
                z_fake_label_pred = model.adversary(z_fake_pred_norms, z_fake_pred_sphere)
                z_fake_label = torch.zeros(z_fake_label_pred.size(),device=z_fake_label_pred.device)
                adv_loss = bce_lossfunc(z_real_label_pred, z_real_label) + bce_lossfunc(z_fake_label_pred, z_fake_label)
                adv_loss = (1.0/2)*adv_loss

                if (epoch >= ring_loss_after) and (epoch <= ring_loss_max):
                    ring_loss = ring_lossfunc(z_aug_pred_norms)
                else:
                    ring_loss = torch.tensor(0.0)


                # Get reconstructions
                x_real_recon, x_real_recon_feats = model.decode(z_real_pred_norms, z_real_pred_sphere)
                x_fake_recon, x_fake_recon_feats = model.decode(z_fake_pred_norms, z_fake_pred_sphere)
                x_aug_recon, x_aug_recon_feats = model.decode(z_aug_pred_norms, z_aug_pred_sphere)

                x_real_recon = torch.tanh(x_real_recon)
                x_fake_recon = torch.tanh(x_fake_recon)
                x_aug_recon = torch.tanh(x_aug_recon)


                # Reconstruction losses
                recon_loss = recon_lossfunc(x_real_recon, x_real) + recon_lossfunc(x_fake_recon, x_fake) + recon_lossfunc(x_aug_recon, x_aug)
                recon_loss = (1.0/3)*recon_loss

                if use_features:
                    recon_feat_loss = l2_lossfunc(x_real_recon_feats, x_real_feats) + l2_lossfunc(x_fake_recon_feats, x_fake_feats) + l2_lossfunc(x_aug_recon_feats, x_aug_feats)
                    recon_feat_loss = (1.0/2)*recon_feat_loss.mean()
                else:
                    recon_feat_loss = torch.tensor(0.0)

                # Tracking the losses
                loss_str = "z_norm_loss: {0}, z_cosine_loss: {1}, recon_loss: {2}\nrecon_feat_loss: {3}, adv_loss: {4}, ring_loss: {5}".format(z_norm_loss.item(), z_cosine_loss.item(), recon_loss.item(), recon_feat_loss.item(), adv_loss.item(), ring_loss.item())
                # Tracking the z norms
                norm_str = "real_norms: {0}, fake_norms: {1}".format(z_real_pred_norms.mean().item(), z_fake_pred_norms.mean().item())
                loss_str = loss_str + '\n' + norm_str
                loss_str += '\n'+str(z_aug_pred_norms[:20].detach().cpu())


                # Add all losses and step optimizer
                total_loss = lmbda_norm*z_norm_loss + lmbda_cosine*z_cosine_loss + \
                             lmbda_recon*recon_loss + lmbda_feat*recon_feat_loss + \
                             (-1)*lmbda_adv*adv_loss + lmbda_ring*ring_loss

                model.zero_grad()
                total_loss.backward()
                model.step_optim()



            elif phase==1:
                z_real_pred_norms, z_real_pred_sphere = model.encode(x_real, features=x_real_feats)
                z_fake_pred_norms, z_fake_pred_sphere = model.encode(x_fake, features=x_fake_feats)

                z_real_label_pred = model.adversary(z_real_pred_norms, z_real_pred_sphere)
                z_real_label = torch.ones(z_real_label_pred.size(),device=z_real_label_pred.device)
                z_fake_label_pred = model.adversary(z_fake_pred_norms, z_fake_pred_sphere)
                z_fake_label = torch.zeros(z_fake_label_pred.size(),device=z_fake_label_pred.device)
                adv_loss = bce_lossfunc(z_real_label_pred, z_real_label) + bce_lossfunc(z_fake_label_pred, z_fake_label)
                adv_loss = (1.0/2)*adv_loss

                model.zero_grad(mode='adversary')
                adv_loss.backward()
                model.step_optim(mode='adversary')

                loss_str = "adv_loss: {0}".format(adv_loss.item())

            if (i>1) and ((i%100 == 0) or (i%100 == 1)):
                print(i)
                print(loss_str)


            phase = 1-phase



