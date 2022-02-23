from ae_models import *
from utils import EasyDict as edict





class Feature_Encoder(nn.Module):
    def __init__(self):
        super().__init__(self, linear=False, nfeats=1000)

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
    def __init__(self):
        super().__init__(self, linear=False, nfeats=1000)

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
        self.modules.encoder = Encoder(32, ncolor=3, lean=False, very_lean=False, all_linear=False, add_linear=False).to(device)
        self.modules.decoder = Decoder(32, ncolor=3, lean=False, very_lean=False, all_linear=False, add_linear=False).to(device)
        self.modules.e2z = Phi(num_out=513).to(device) # 513 due to separate norm and direction
        self.modules.z2e = Phi(num_in=513).to(device)
        self.modules.adversary = Small_Classifier(linear=False).to(device)

        
        if use_features:
            self.modules.feature_encode = Feature_Encoder.to(device)
            self.modules.feature_decode = Feature_Decoder.to(device)
        self.use_features = use_features


        self.optimizers = edict()
        for network in self.modules:
            self.optimizers[network] = torch.optim.Adam(self.modules[network].parameters(), lr=global_lr)

           


    def train(self):
        for model in self.modules:
            model.train()

    def eval(self):
        for model in self.modules:
            model.eval()


    # Think about how to do the normalization
    def encode(self, x, features=None):
        e = self.modules.encoder(x)

        if self.use_features:
            e = self.modules.feature_encode(e,features)

        z = self.modules.e2z(e)

        z_norms, z_normalized = torch.split(z, [1, 512])
        z_normalized = z_normalized / torch.norm(z_normalized, dim=1)


#        z_norms = torch.norm(z,dim=1)
#        z_normalized = z / z_norms
        return z_norms, z_normalized
        

    def decode(self, z_norms, z_normalized):
        z_concat = torch.cat([z_norms, z_normalized],dim=1)
        e = self.modules.z2e(z_concat)

        if self.use_features:
            e, features = self.modules.feature_decode(e)
        else:
            features = None

        x = self.modules.decoder(e)
        return x, features
        

    def adversary(self, z_norms, z_normalized):
        z_concat = torch.cat([z_norms, z_normalized],dim=1)
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
        

def train_combined(model, dataloader, n_epochs, use_features=False):
    l1_lossfunc = torch.nn.MSELoss()
    l2_lossfunc = torch.nn.L1Loss()
    bce_lossfunc = torch.nn.BCEWithLogitsLoss()
   
    def ring_pressure():
        pass
    def z_regularization():
        pass


    phase = 0
    for epoch in range(n_epochs):
        print("epoch",epoch)
        for i, batch in enumerate(dataloader):
            x_real = batch['real'][0].cuda()
            x_fake = batch['fake'][0].cuda()
            z_fake = batch['fake'][1].cuda()
            z_fake_norms = torch.norm(z_fake,dim=1).detach()
            z_fake_sphere = (z_fake/z_fake_norms).detach()
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
                z_norm_loss = (z_fake_pred_norms - z_fake_norms).square()
                z_cosine_loss = 1 - (z_fake_sphere*z_fake_pred_sphere).sum(dim=1)

                z_real_label_pred = model.adversary(z_real_pred_norms, z_real_pred_sphere)
                z_real_label = torch.ones(z_real_label_pred.size(),device=z_real_label_pred.device)
                z_fake_label_pred = model.adversary(z_fake_pred_norms, z_fake_pred_sphere)
                z_fake_label = torch.zeros(z_fake_label_pred.size(),device=z_fake_label_pred.device)
                adv_loss = bce_lossfunc(z_real_label_pred, z_real_label) + bce_lossfunc(z_fake_label_pred, z_fake_label)

                


                # Get reconstructions
                x_real_recon, x_real_recon_feats = model.decode(z_real_pred_norms, z_real_pred_sphere)
                x_fake_recon, x_fake_recon_feats = model.decode(z_fake_pred_norms, z_fake_pred_sphere)
                x_aug_recon, x_aug_recon_feats = model.decode(z_aug_pred_norms, z_aug_pred_sphere)


                # Reconstruction losses
                recon_loss = l2_lossfunc(x_real_recon, x_real) + l2_lossfunc(x_fake_recon, x_fake) + l2_lossfunc(x_aug_recon, x_aug)

                if use_features:
                    recon_feat_loss = l2_lossfunc(x_real_recon_feats, x_real_feats) + l2_lossfunc(x_fake_recon_feats, x_fake_feats) + l2_lossfunc(x_aug_recon_feats, x_aug_feats)
                else:
                    recon_feat_loss = 0


            elif phase==1:
                z_real_pred_norms, z_real_pred_sphere = model.encode(x_real, features=x_real_feats)
                z_fake_pred_norms, z_fake_pred_sphere = model.encode(x_fake, features=x_fake_feats)

                # Get adversary losses




"""
# Need: Phi model to handle normalized z values
# Ring pressure loss
# Losses: reconstruction, z reconstruction, adversary, ring pressure    

def train_autoencoder(model, dataloader, n_epochs, use_adversary=True, use_features=False):
    #reconstruction_loss = torch.nn.BCEWithLogitsLoss()
    #reconstruction_loss = torch.nn.MSELoss()
    #reconstruction_loss = torch.nn.L1Loss()
    
    l1_lossfunc = torch.nn.MSELoss()
    l2_lossfunc = torch.nn.L1Loss()
    def reconstruction_loss(x,y):
        return l1_lossfunc(x,y)+l2_lossfunc(x,y)
    
    def reg_loss(e):
        return (torch.norm(e.view(e.size(0),-1), dim=1)**2).mean()
    
    
    model.encoder.train()
    model.decoder.train()
    
    if use_adversary:
        model.domain_adversary.train()
        adversary_loss = torch.nn.BCEWithLogitsLoss()
        lmbda_adv = 1
        lmbda_reg = 0.001
    else:
        lmbda_adv = 0
        lmbda_reg = 0

    if use_features:
        model.feature_encode.train()
        model.feature_decode.train()

    
    phase = 0
    for epoch in range(n_epochs): 
        print("Epoch", epoch)
        for i, batch in enumerate(dataloader):
            x_real = batch['real'][0].cuda()
            x_fake = batch['fake'][0].cuda()
            z_fake = batch['fake'][1].cuda()
            x_aug = batch['augmented'][0].cuda()

            if use_features:
                x_real_feats = batch['real'][-2].cuda() # last one is labels
                x_fake_feats = batch['fake'][-2].cuda()
                x_aug_feats = batch['augmented'][-2].cuda()
            
            if use_adversary:
                y_real = torch.zeros(len(x_real),device=x_real.device) #domain labels
                y_fake = torch.ones(len(x_fake), device=x_fake.device) #domain labels
                       
            loss_info = None
            if phase == 0:
                # train encoder and decoder
                enc_x_real = model.encoder(x_real)
                enc_x_fake = model.encoder(x_fake)
                enc_x_aug = model.encoder(x_aug)
 
                if use_features:
                    enc_x_real = enc_x_real + model.feature_encode(x_real_feats)
                    enc_x_fake = enc_x_fake + model.feature_encode(x_fake_feats)
                    enc_x_aug = enc_x_aug + model.feature_encode(x_aug_feats)
                    
                    real_add, recon_x_real_feats = model.feature_decode(enc_x_real)
                    fake_add, recon_x_fake_feats = model.feature_decode(enc_x_fake)
                    aug_add, recon_x_aug_feats = model.feature_decode(enc_x_aug)

                    loss_real_feats = l2_lossfunc(recon_x_real_feats, x_real_feats)
                    loss_fake_feats = l2_lossfunc(recon_x_fake_feats, x_fake_feats)
                    loss_aug_feats = l2_lossfunc(recon_x_aug_feats, x_aug_feats)
                else:
                    recon_real_feats = 0
                    recon_fake_feats = 0
                    recon_aug_feats = 0
                    
                
                if use_adversary:
                    predicted_real_domains = model.domain_adversary(enc_x_real)
                    predicted_fake_domains = model.domain_adversary(enc_x_fake)
                    adv_loss = (1.0/2)*(adversary_loss(predicted_real_domains,y_real)+adversary_loss(predicted_fake_domains, y_fake))
                    regularizing_loss = (1.0/3)*(reg_loss(enc_x_real)+reg_loss(enc_x_fake) + reg_loss(enc_x_aug))
                else:
                    adv_loss = 0
                    regularizing_loss = 0


                if use_features:
                    # idea is to undo adding the features
                    enc_x_real = enc_x_real + real_add
                    enc_x_fake = enc_x_fake + fake_add
                    enc_x_aug = enc_x_aug + aug_add


                recon_real = torch.tanh(model.decoder(enc_x_real)) #add a variant of tanh?
                recon_fake = torch.tanh(model.decoder(enc_x_fake))
                recon_aug = torch.tanh(model.decoder(enc_x_aug))
                
                recon_loss = (1.0/3)*(reconstruction_loss(recon_real, x_real) + reconstruction_loss(recon_fake, x_fake) + reconstruction_loss(recon_aug, x_aug))
                feats_loss = (1.0/3)*(loss_real_feats + loss_fake_feats + loss_aug_feats)

                
                total_loss = recon_loss - lmbda_adv*adv_loss +lmbda_reg*regularizing_loss + feats_loss
                #total_loss = recon_loss#+regularizing_loss
                
                if use_features:
                    model.feature_encode.zero_grad()
                    model.feature_decode.zero_grad()

                model.encoder.zero_grad()
                model.decoder.zero_grad()
                total_loss.backward()
                model.encoder_optim.step()
                model.decoder_optim.step()

                if use_features:
                    model.feature_encode_optim.step()
                    model.feature_decode_optim.step()
                

                lossinfo = "recon loss = {0}".format(recon_loss.item())
                if use_adversary:
                    lossinfo += ", adv loss = {0}, reg loss = {1}".format(adv_loss.item(), regularizing_loss.item())
                if use_features:
                    lossinfo += ", feats loss = {0}".format(feats_loss.item())
                
                
            if phase == 1:
                # Train adversary
               
                enc_x_real = model.encoder(x_real)
                enc_x_fake = model.encoder(x_fake)
                if use_features:
                    enc_x_real += model.feature_encode(x_real_feats)
                    enc_x_fake += model.feature_encode(x_fake_feats)

                predicted_real_domains = model.domain_adversary(enc_x_real)
                predicted_fake_domains = model.domain_adversary(enc_x_fake)
                adv_loss = (1.0/2)*(adversary_loss(predicted_real_domains,y_real)+adversary_loss(predicted_fake_domains, y_fake))
                               
                model.domain_adversary.zero_grad()
                adv_loss.backward()
                model.domain_adversary_optim.step()
    
                lossinfo = "adv loss = {0}".format(adv_loss.item())
    
    
            if (i > 1) and (i-phase)%100 == 0:
                print(i, lossinfo)
    
            if use_adversary:
                phase = 1-phase
    
    


# 3 is "significance factor" (in 3**4 express)
def pressure_loss(x, lmbda_1 = 0.5*(512**(1.5)), lmbda_2 = 1/(2*(512**0.5)*(3**4)), eps = 1e-5):
    square_norms = x.square().sum(dim=1)
    outward_pressure = lmbda_1 / (square_norms + eps)
    inward_pressure = lmbda_2 * square_norms
    total_loss = (inward_pressure+outward_pressure).mean()
    return total_loss  

def ring_pressure_loss(x, r2=512, w=31, significance_factor = 3, eps=1e-5):
    ring_lmbda = (w**2) / (2*(r2-w))
    reg_lmbda = ring_lmbda / ((significance_factor*w)**2)
    x2 = x.square().sum(dim=1)
    ring = ring_lmbda / ((x2-r2).abs() + eps)
    reg = reg_lmbda*x2
    total_loss = (ring+reg).mean()
    return total_loss


def mse_loss(x,y):
    return torch.nn.functional.mse_loss(x,y)

def clf_loss(x,y):
    return torch.nn.functional.binary_cross_entropy_with_logits(x,y)



def train_invertible(model, dataloader, n_epochs, use_adversary=False, use_friend=False, print_every=200, pressure='ring'):
    lossfunc = mse_loss
    lmbda_a = 1
    lmbda_f = 1
    lmbda_z_fake = 1
    lmbda_e_fake = 1

    lmbda_aug_cycle = 1
    lmbda_cycle = 1

    if pressure == 'ring':
        print("using ring pressure")
        p_lossfunc = ring_pressure_loss
        lmbda_p = 1
    else:
        p_lossfunc = pressure_loss
        lmbda_p = 3

    model.e2z.train()
    model.z2e.train()


    adversary_loss = None
    friend_lossfunc = None
    if use_adversary:
        model.adversary.train()
        adversary_loss = clf_loss
    if use_friend:
        model.friend.train()
        friend_lossfunc = clf_loss


    lossinfo = None
    phase = 0
    for n in range(n_epochs):
        print("Epoch", n)
        if n < 10:
            print("no adversary, no pressure, no cycle, only bidirectinal targets")
            lmbda_a = 0
            lmbda_p = 0
            lmbda_aug_cycle = 0
            lmbda_cycle = 0.05
        elif n < 20:
            print("add cycle loss")
            lmbda_a = 0
            lmbda_p = 0
            lmbda_aug_cycle = 0.05
            lmbda_cycle = 1
        else:
            if (n // 5)%2 == 0:
                # Use adversary but no pressure
                print("adversary, no pressure, aug cycle")
                lmbda_a = 1 
                lmbda_p = 0
                lmbda_aug_cycle = 1
                lmbda_cycle = 1

            else:
                print("pressure, no adversary, aug cycle")
                lmbda_a = 0
                lmbda_p = 0.05
                lmbda_aug_cycle = 1
                lmbda_cycle = 1


        for i, batch in enumerate(dataloader):
            z_fake, e_fake, _ = batch['fake']
            e_real, _ = batch['real']
            e_aug, _ = batch['augmented']
            
            
            z_fake = z_fake.cuda()
            e_fake = e_fake.cuda()
            e_real = e_real.cuda()
            e_aug = e_aug.cuda()

            y_real = None
            y_fake = None
            y_aug = None
            
            if use_adversary:
                y_real = torch.ones(len(e_real), device=e_real.device)
                y_fake = torch.zeros(len(e_fake), device=e_fake.device)
                y_aug = torch.zeros(len(e_aug), device=e_aug.device)
            

            if phase == 0:
                z_real_pred = model.e2z(e_real)
                z_fake_pred = model.e2z(e_fake)
                z_aug_pred = model.e2z(e_aug)




                if use_adversary:
                    y_real_pred = model.adversary(z_real_pred)
                    y_fake_pred = model.adversary(z_fake_pred)
                    adv_loss = 0.5*(adversary_loss(y_real_pred, y_real) + adversary_loss(y_fake_pred, y_fake))
                else:
                    adv_loss = 0


                if use_friend:
                    y_real_pred = model.friend(z_real_pred)
                    y_aug_pred = model.friend(z_aug_pred)
                    friend_loss = 0.5*(friend_lossfunc(y_real_pred, y_real) + friend_lossfunc(y_aug_pred, y_aug))
                else:
                    friend_loss = 0


                
                e_real_cycle = model.z2e(z_real_pred)
                e_fake_cycle = model.z2e(z_fake_pred)
                e_aug_cycle =model.z2e(z_aug_pred)

                
                e_fake_pred = model.z2e(z_fake)
                z_fake_cycle = model.e2z(e_fake_pred)
                
                # z2e loss
                e_fake_loss = lossfunc(e_fake_pred, e_fake)
                z_fake_cycle_loss = lossfunc(z_fake_cycle, z_fake)

                #e2z loss
                z_fake_loss = lossfunc(z_fake_pred, z_fake)

                # cycle losses
                e_fake_cycle_loss = lossfunc(e_fake_cycle, e_fake)
                e_real_cycle_loss = lossfunc(e_real_cycle, e_real)
                e_aug_cycle_loss = lossfunc(e_aug_cycle, e_aug)

                # pressure losses
                z_aug_pressure_loss = p_lossfunc(z_aug_pred)
                
                total_loss = lmbda_z_fake*z_fake_loss + lmbda_e_fake*e_fake_loss + \
                               lmbda_cycle*(e_fake_cycle_loss + e_real_cycle_loss + z_fake_cycle_loss) + \
                               lmbda_aug_cycle*e_aug_cycle_loss - lmbda_a*adv_loss +\
                               lmbda_f*friend_loss + lmbda_p*z_aug_pressure_loss

                if use_friend:
                    model.friend.zero_grad()
                model.z2e.zero_grad()
                model.e2z.zero_grad()
                total_loss.backward()
                model.z2e_optim.step()
                model.e2z_optim.step()
                if use_friend:
                    model.friend_optim.step()


                if i > 1 and (i%print_every == 0 or (i+1)%print_every==0):
                    lossinfo = "  {5}: z_fake_loss={0}, e_fake_cycle_loss={1}, e_real_cycle_loss={2}, e_aug_cycle_loss={3}, z_aug_pressure_loss={4}".format(z_fake_loss.item(), e_fake_cycle_loss.item(), e_real_cycle_loss.item(), e_aug_cycle_loss.item(), z_aug_pressure_loss.item(), i)
    
                    if use_adversary:
                        lossinfo += ", adv_loss={0}".format(adv_loss.item())
                    if use_friend:
                        lossinfo += ", friend_loss={0}".format(friend_loss.item()) 
    
                    z_norms = {'real':torch.norm(z_real_pred,dim=1).square().mean().item(),
                               'fake':torch.norm(z_fake_pred,dim=1).square().mean().item(),
                               'aug':torch.norm(z_aug_pred,dim=1).square().mean().item()}
    
    
    
                    lossinfo += '\n' + str(z_norms)
                

            elif phase == 1:
                z_real_pred = model.e2z(e_real)
                z_fake_pred = model.e2z(e_fake)
                
                y_real_pred = model.adversary(z_real_pred)
                y_fake_pred = model.adversary(z_fake_pred)

                adv_loss = 0.5*(adversary_loss(y_real_pred, y_real) + adversary_loss(y_fake_pred, y_fake))
                model.adversary.zero_grad()
                adv_loss.backward()
                model.adversary_optim.step()

                lossinfo = "adv loss = {0}".format(adv_loss.item())
               
                

            if i>1 and (i%print_every==0 or (i+1)%print_every ==0):
                print(lossinfo)

            if use_adversary:
                phase = 1-phase
"""
