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






class Combined_Autoencoder:
    def __init__(self, global_lr=0.001, use_features=True, device='cuda:0'):

        self.modules = edict()
        self.modules.encoder = Encoder(32, ncolor=3, lean=False, very_lean=False, all_linear=False, add_linear=False).to(device)
        self.modules.decoder = Decoder(32, ncolor=3, lean=False, very_lean=False, all_linear=False, add_linear=False).to(device)
        self.modules.e2z = Phi().to(device)
        self.modules.z2e = Phi().to(device)
        self.modules.adversary = Small_Classifier(linear=False).to(device)

        
        if use_features:
            self.modules.feature_encode = Feature_Encoder.to(device)
            self.modules.feature_decode = Feature_Decoder.to(device)
        self.use_features = use_features


        self.optimizers = edict()
        for network in self.modules:
            self.optimizers[network] = torch.optim.Adam(self.modules[network].parameters(), lr=global_lr)

           


    def encode(self, x, features=None):
        e = self.modules.encoder(x)

        if self.use_features:
            e = self.feature_encode(e,features)

        z = self.modules.e2z(e)
        z_norms = torch.norm(z,dim=1)
        z_normalized = z / z_norms
        return z_norms, z_normalized
        

    def decode(self, z_norms, z_normalized):
        z_concat = torch.cat([z_norms, z_normalized],dim=1)
        e = self.modules.z2e(z_concat)

        if self.use_features:
            e, features = self.feature_decode(e)
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
        


# Need: Phi model to handle normalized z values
# Ring pressure loss
# Losses: reconstruction, z reconstruction, adversary, ring pressure    
