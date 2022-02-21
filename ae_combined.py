from ae_models import *
from utils import EasyDict as edict


class Combined_Autoencoder:
    def __init__(self, global_lr=0.001):

        self.modules = edict()
        self.modules.encoder = Encoder(32, ncolor=3, lean=False, very_lean=False, all_linear=False, add_linear=False).cuda()
        self.modules.decoder = Decoder(32, ncolor=3, lean=False, very_lean=False, all_linear=False, add_linear=False).cudam()
        self.modules.e2z = Phi().cuda()
        self.modules.z2e = Phi().cuda()
        self.modules.adversary = Small_Classifier(linear=False).cuda()


        self.optimizers = edict()
        for network in self.modules:
            self.optimizers[network] = torch.optim.Adam(self.modules[network].parameters(), lr=global_lr)
            


    def encode(self, x):
        e = self.modules.encoder(x)
        z = self.modules.e2z(e)
        z_norms = torch.norm(z,dim=1)
        z_normalized = z / z_norms
        return z_norms, z_normalized
        
    def decode(self, z_norms, z_normalized):
        z_concat = torch.cat([z_norms, z_normalized],dim=1)
        e = self.modules.z2e(z_concat)
        x = self.modules.decoder(e)
        return x
        
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
        
    
