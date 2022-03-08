from ae_models import *
from ae_data import *
from ae_method import view_tensor_images
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
    def __init__(self, global_lr=0.001, use_features=True, device='cuda:0', use_simple_nets=False, use_linear_nets = False, use_layer_norm = False):

        self.modules = edict()


        self.use_simple_nets = use_simple_nets
        self.use_linear_nets = use_linear_nets

        if use_simple_nets:
            if use_linear_nets:
                activation = nn.LeakyReLU(0.2,inplace=True)
            else:
                activation = nn.Identity()


            self.modules.encoder = nn.Sequential(
                nn.Linear(3072,1024),
                activation,
                nn.Linear(1024,512)
            ).cuda()

            self.modules.decoder = nn.Sequential(
                nn.Linear(512,1024),
                activation,
                nn.Linear(1024,3072)
            ).cuda()

        else:
            if use_linear_nets:
                all_linear = True
                very_lean = True
                add_linear = True
                lean=False
            else:
                all_linear = False
                add_linear = False
                very_lean = False
                lean = True
            self.modules.encoder = Encoder(32, ncolors=3, lean=lean, very_lean=very_lean, all_linear=all_linear, add_linear=add_linear, use_layer_norm = use_layer_norm).to(device)
            self.modules.decoder = Decoder(32, ncolors=3, lean=lean, very_lean=very_lean, all_linear=all_linear, add_linear=add_linear, use_layer_norm=use_layer_norm).to(device)



        if use_layer_norm:
            norm_layer = nn.LayerNorm
        else:
            norm_layer = nn.BatchNorm1d

        self.modules.e2z = Phi(num_out=513, norm_layer=norm_layer).to(device) # 513 due to separate norm and direction
        self.modules.z2e = Phi(num_in=513, norm_layer=norm_layer).to(device)
        self.modules.adversary = Small_Classifier(linear=False, in_feats=513).to(device)

        
        if use_features:
            self.modules.feature_encode = Feature_Encoder(linear=use_linear_nets).to(device)
            self.modules.feature_decode = Feature_Decoder(linear=use_linear_nets).to(device)
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
        if hasattr(self, 'use_simple_nets') and self.use_simple_nets:
            x = x.reshape(x.size(0), -1)

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
        
        if (not hasattr(self, 'use_simple_nets')) or (not self.use_simple_nets):
            e = e.reshape(e.size(0), 32, 4, 4)

        x = self.modules.decoder(e)

        if hasattr(self, 'use_simple_nets') and self.use_simple_nets:
            x = x.reshape(x.size(0), 3, 32, 32)

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


def simple_ring_lossfunc(x, dim=512, significance=3, sigma=0.7):
    T = dim**(0.5)
    S = significance*sigma
    loss = torch.min(x - (T-S), -x + (T+S)).clamp(0)
    return loss.mean()

def simpler_ring_lossfunc(x, dim=512, significance=3, sigma=0.7):
    T = dim**(0.5)
    S = significance*sigma
    loss = (-x + (T+S)).clamp(0)
    return loss.mean()



def train_combined(model, dataloader, n_epochs, use_features=False, ring_loss_after=10, ring_loss_max=10000,
                   lmbda_norm = 1, lmbda_cosine=1, lmbda_recon=1, lmbda_feat=1, lmbda_adv=1,
                   lmbda_ring=1, significance=3, use_simple_ring_loss = False, use_simpler_ring_loss = False,
                   use_augmented = True, print_every=100,
                   use_adversary = True
                   ):

    model.train()

    l1_lossfunc = torch.nn.MSELoss()
    l2_lossfunc = torch.nn.L1Loss()
    bce_lossfunc = torch.nn.BCEWithLogitsLoss()

    if use_simple_ring_loss:
        apply_ring_loss = simple_ring_lossfunc
    elif use_simpler_ring_loss:
        print("using simpler ring loss")
        apply_ring_loss = simpler_ring_lossfunc
    else:
        apply_ring_loss = ring_lossfunc

    def recon_lossfunc(x,y):
        return l1_lossfunc(x,y)+l2_lossfunc(x,y)

    phase = 0
    for epoch in range(n_epochs):
        print("***************************epoch",epoch, "*************************")
        for i, batch in enumerate(dataloader):
            x_real = batch['real'][0].cuda()
            x_fake = batch['fake'][1].cuda()
            z_fake = batch['fake'][0].cuda()
            z_fake_norms = torch.norm(z_fake,dim=1).detach()
            z_fake_sphere = (z_fake/z_fake_norms.unsqueeze(1)).detach()

            if use_augmented:
                x_aug = batch['augmented'][0].cuda()
            else:
                x_aug = None

            if use_features:
                x_real_feats = batch['real'][-2].cuda()
                x_fake_feats = batch['fake'][-2].cuda()

                if use_augmented:
                    x_aug_feats = batch['augmented'][-2].cuda()
                else:
                    x_aug_feats = None

            else:
                x_real_feats = None
                x_fake_feats = None
                x_aug_feats = None

            loss_info = None
            if phase == 0: #encoder/decoder
                z_real_pred_norms, z_real_pred_sphere = model.encode(x_real, features=x_real_feats)
                z_fake_pred_norms, z_fake_pred_sphere = model.encode(x_fake, features=x_fake_feats)

                if use_augmented:
                    z_aug_pred_norms, z_aug_pred_sphere = model.encode(x_aug, features=x_aug_feats)
                else:
                    z_aug_pred_norms, z_aug_pred_sphere = None, None
                
                # Z losses and adversary losses
                z_norm_loss = (z_fake_pred_norms - z_fake_norms).square().mean()
                z_cosine_loss = 1 - (z_fake_sphere*z_fake_pred_sphere).sum(dim=1).mean()
                # use l2 instead of cosine? or does it not matter?

                if use_adversary:
                    z_real_label_pred = model.adversary(z_real_pred_norms, z_real_pred_sphere)
                    z_real_label = torch.ones(z_real_label_pred.size(),device=z_real_label_pred.device)
                    z_fake_label_pred = model.adversary(z_fake_pred_norms, z_fake_pred_sphere)
                    z_fake_label = torch.zeros(z_fake_label_pred.size(),device=z_fake_label_pred.device)
                    adv_loss = bce_lossfunc(z_real_label_pred, z_real_label) + bce_lossfunc(z_fake_label_pred, z_fake_label)
                    adv_loss = (1.0/2)*adv_loss
                else:
                    adv_loss = torch.tensor(0.0)

                if use_augmented and (epoch >= ring_loss_after) and (epoch <= ring_loss_max):
                    ring_loss = apply_ring_loss(z_aug_pred_norms, significance=significance)
                else:
                    ring_loss = torch.tensor(0.0)


                # Get reconstructions
                x_real_recon, x_real_recon_feats = model.decode(z_real_pred_norms, z_real_pred_sphere)
                x_real_recon = torch.tanh(x_real_recon)

                x_fake_recon, x_fake_recon_feats = model.decode(z_fake_pred_norms, z_fake_pred_sphere)
                x_fake_recon = torch.tanh(x_fake_recon)


                if use_augmented:
                    x_aug_recon, x_aug_recon_feats = model.decode(z_aug_pred_norms, z_aug_pred_sphere)
                    x_aug_recon = torch.tanh(x_aug_recon)
                else:
                    x_aug_recon = None


                # Reconstruction losses
                recon_loss = recon_lossfunc(x_real_recon, x_real) + recon_lossfunc(x_fake_recon, x_fake) 
                if use_augmented:
                    recon_loss += recon_lossfunc(x_aug_recon, x_aug)
                recon_loss = (1.0/3)*recon_loss


                if use_features:
                    recon_feat_loss = l2_lossfunc(x_real_recon_feats, x_real_feats) + l2_lossfunc(x_fake_recon_feats, x_fake_feats) 
                    if use_augmented:
                        recon_feat_loss += l2_lossfunc(x_aug_recon_feats, x_aug_feats)
                    recon_feat_loss = (1.0/2)*recon_feat_loss.mean()
                else:
                    recon_feat_loss = torch.tensor(0.0)


                if (i>1) and ((i//2)%print_every == 0):
                    # Tracking the losses
                    loss_str = "z_norm_loss: {0}, z_cosine_loss: {1}, recon_loss: {2}\nrecon_feat_loss: {3}, adv_loss: {4}, ring_loss: {5}".format(z_norm_loss.item(), z_cosine_loss.item(), recon_loss.item(), recon_feat_loss.item(), adv_loss.item(), ring_loss.item())
                    # Tracking the z norms
                    norm_str = "real_norms: {0}, fake_norms: {1}".format(z_real_pred_norms.mean().item(), z_fake_pred_norms.mean().item())
                    loss_str = loss_str + '\n' + norm_str

                    if use_augmented:
                        loss_str += '\nz_aug_norm_diffs = {0}, z_aug_norm_means={1}'.format((z_aug_pred_norms-512**0.5).abs().mean().item(), z_aug_pred_norms.mean().item())
                    #loss_str += '\n'+str(z_aug_pred_norms[:20].detach().cpu())

                    print(i)
                    print(loss_str)


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

#                loss_str = "adv_loss: {0}".format(adv_loss.item())



            if use_adversary:
                phase = 1-phase




### Extracting info ####


# View the images corresponding to the top and bottom values
def view_top_and_bottom(values, ims, title=''):
    assert(len(values) == len(ims))
    print('**************'+title+'****************')
    inds = torch.argsort(values)
    top_inds = inds[-64:]
    bottom_inds = inds[:64].flip(0)
    print("Top images")
    view_tensor_images(ims[top_inds])
    print("Bottom images")
    view_tensor_images(ims[bottom_inds])



def extract_probabilities_combined(model_path, model_name_prefix, data_cfg):
    print("In extract probs combined")
    multi_loader = get_dataloaders(data_cfg, 'prob_stage')   


    
    model = pickle.load(open(data_cfg.prob_stage.model_file,'rb'))
    model.eval()    
    
    

    from model_functions import jacobian, log_priors

    
    stats = edict()
    #for name, dataloader in [("class_0", class_0_dataloader), ("class_1", class_1_dataloader),("fake_class_1",fake_class_1_dataloader)]:
        
    for name in multi_loader.keys():
        dataloader = multi_loader.loaders[name]
        
        print(name)
        #e_codes = []
        e_differences = []
        e_norms = []
        z_norms = []
        logpriors = []
        z2e_jacobian_probs = []
        e2z_jacobian_probs = []
        total_z2e_probs = []
        total_e2z_probs = []
        for i, batch in enumerate(dataloader):
            print("  {0} of {1}".format(i,len(dataloader)))
            ims = batch[0]
            ims = ims.cuda()

            if hasattr(model, 'use_simple_nets') and model.use_simple_nets:
                ims = ims.reshape(ims.size(0),-1)
            e_c = model.modules.encoder(ims)
            e_c = e_c.reshape(e_c.size(0),-1)

            if model.use_features:
                features = batch[1]
                features = features.cuda()
                e_c = model.modules.feature_encode(e_c,features)
           

            # Predict z and convert it back into 512-dimensional form 
            e_c = e_c.detach()
            e_c.requires_grad_(True)
            z_predicted = model.modules.e2z(e_c)
            z_norm = z_predicted[:,0].unsqueeze(1)
            z_predicted = z_norm * (z_predicted[:,1:] / torch.norm(z_predicted[:,1:],dim=1).unsqueeze(1))
            assert(z_predicted.size(1) == 512)           
            forward_jacobians = -jacobian(e_c, z_predicted).detach().cpu()
            e2z_jacobian_probs.append(forward_jacobians)

            
            # Save the predicted norms
            z_norm = z_norm.detach().cpu().squeeze(1)
            z_norms.append(z_norm)

            # detach the 512-dim z and compute log priors
            z_predicted = z_predicted.detach()
            z_log_priors = log_priors(z_predicted)
            logpriors.append(z_log_priors)

            # recompute the norm and scaled direction for gradient purposes, put into 513-dim vector
            z_predicted.requires_grad_(True)
            z_norm_recomp = torch.norm(z_predicted,dim=1).unsqueeze(1)
            z_directions = z_predicted / z_norm_recomp
            z_predicted_recomp = torch.cat([z_norm_recomp, z_directions], dim=1)
            assert(z_predicted_recomp.size(1) == 513)
    
            e_reconstructed = model.modules.z2e(z_predicted_recomp)
            inv_jacobians = jacobian(z_predicted, e_reconstructed).detach().cpu()
            z2e_jacobian_probs.append(inv_jacobians)
            
            diffs = (e_c - e_reconstructed).detach().cpu()
            e_differences.append(diffs)
            e_norms.append(torch.norm(diffs,dim=1))
            
            
            total_z2e_probs.append(z_log_priors+inv_jacobians)
            total_e2z_probs.append(z_log_priors+forward_jacobians)
        
        stats[name] = edict()
        #stats[name].e_codes = torch.cat(e_codes)
        stats[name].e_differences = torch.cat(e_differences)
        stats[name].e_norms = torch.cat(e_norms)
        stats[name].log_priors = torch.cat(logpriors)
        stats[name].z2e_jacobian_probs = torch.cat(z2e_jacobian_probs)
        stats[name].e2z_jacobian_probs = torch.cat(e2z_jacobian_probs)
        stats[name].total_z2e_probs = torch.cat(total_z2e_probs)
        stats[name].total_e2z_probs = torch.cat(total_e2z_probs)
        stats[name].z_norms = torch.cat(z_norms)
        
        
    
    # 1. Encode the real data, detach encodings from graph
    # 2. Run encodings through e2z and z2e, get logprobs and log priors
    # 3. Plot (3 graphs: jacobian dets, priors, and combined)
    
    print(stats.keys())
    for key in stats:
        print(stats[key].keys())
        
    save_path = os.path.join(model_path, model_name_prefix+'_extracted.pkl')
    pickle.dump(stats, open(save_path,'wb'))    





def view_extracted_probabilities_combined(model_path, model_name_prefix, data_cfg, aug_label="Real planes"):
    # note the off-manifold scores are norms here, not squared norms
    
    #data = pickle.load(open('./models/autoencoder/extracted_info_exp_4.pkl', 'rb'))
    data = pickle.load(open(data_cfg.plot_stage.prob_sample_file,'rb'))

    ######### View top and bottom of each ###########3
    real_images = torch.load(data_cfg.visualize_stage.real)[1]['test']['images']
    fake_images = torch.load(data_cfg.visualize_stage.fake)[1]['test']['images']
    aug_images = torch.load(data_cfg.visualize_stage.augmented)[0]['test']['images']

    real_znorms = data['real']['z_norms']
    real_e2z = data['real']['e2z_jacobian_probs']
    real_z2e = data['real']['z2e_jacobian_probs']

    fake_znorms = data['fake']['z_norms']
    fake_e2z = data['fake']['e2z_jacobian_probs']
    fake_z2e = data['fake']['z2e_jacobian_probs']

    aug_znorms = data['augmented']['z_norms']
    aug_e2z = data['augmented']['e2z_jacobian_probs']
    aug_z2e = data['augmented']['z2e_jacobian_probs']

    view_top_and_bottom(real_znorms, real_images, 'Real cars, z norms')
    view_top_and_bottom(real_e2z, real_images, 'Real cars, e2z')
    view_top_and_bottom(real_z2e, real_images, 'Real cars, z2e')

    view_top_and_bottom(fake_znorms, fake_images, 'Fake cars, z norms')
    view_top_and_bottom(fake_e2z, fake_images, 'Fake cars, e2z')
    view_top_and_bottom(fake_z2e, fake_images, 'Fake cars, z2e')

    view_top_and_bottom(aug_znorms, aug_images, aug_label+', z norms')
    view_top_and_bottom(aug_e2z, aug_images, aug_label+', e2z')
    view_top_and_bottom(aug_z2e, aug_images, aug_label+', z2e')

    ###################

    
    
    plt.title("logpriors")
    plt.hist(data.real.log_priors.numpy(), bins=50, density=True, alpha=0.3, label="Real cars")
    plt.hist(data.fake.log_priors.numpy(), bins=50, density=True, alpha=0.3, label="Fake cars")    
    plt.hist(data.augmented.log_priors.numpy(), bins=50, density=True, alpha=0.3, label=aug_label)
    plt.legend()
    plt.show()
    
    plt.title("e2z jacobians")
    plt.hist(data.real.e2z_jacobian_probs.numpy(), bins=50, density=True, alpha=0.3, label="Real cars")
    plt.hist(data.fake.e2z_jacobian_probs.numpy(), bins=50, density=True, alpha=0.3, label="Fake cars")    
    plt.hist(data.augmented.e2z_jacobian_probs.numpy(), bins=50, density=True, alpha=0.3, label=aug_label)
    plt.legend()
    plt.show()    
    
    
    plt.title("e2z combined")
    plt.hist(data.real.total_e2z_probs.numpy(), bins=50, density=True, alpha=0.3, label="Real cars")
    plt.hist(data.fake.total_e2z_probs.numpy(), bins=50, density=True, alpha=0.3, label="Fake cars")    
    plt.hist(data.augmented.total_e2z_probs.numpy(), bins=50, density=True, alpha=0.3, label=aug_label)
    plt.legend()
    plt.show()       
    
    plt.title("z2e jacobians")
    plt.hist(data.real.z2e_jacobian_probs.numpy(), bins=50, density=True, alpha=0.3, label="Real cars")
    plt.hist(data.fake.z2e_jacobian_probs.numpy(), bins=50, density=True, alpha=0.3, label="Fake cars")    
    plt.hist(data.augmented.z2e_jacobian_probs.numpy(), bins=50, density=True, alpha=0.3, label=aug_label)
    plt.legend()
    plt.show()     
    
    
    plt.title("z2e combined")
    plt.hist(data.real.total_z2e_probs.numpy(), bins=50, density=True, alpha=0.3, label="Real cars")
    plt.hist(data.fake.total_z2e_probs.numpy(), bins=50, density=True, alpha=0.3, label="Fake cars")    
    plt.hist(data.augmented.total_z2e_probs.numpy(), bins=50, density=True, alpha=0.3, label=aug_label)
    plt.legend()
    plt.show()     
    
    plt.title("z norms")
    plt.hist(data.real.z_norms.cpu().numpy(), bins=50, density=True, alpha=0.3, label="Real cars")
    plt.hist(data.fake.z_norms.cpu().numpy(), bins=50, density=True, alpha=0.3, label="Fake cars")    
    plt.hist(data.augmented.z_norms.cpu().numpy(), bins=50, density=True, alpha=0.3, label=aug_label)
    plt.legend()
    plt.show()      





def visualize_model_combined(model_path, model_name_prefix, data_cfg, class_constant_stylegan=1):
    multi_loader = get_dataloaders(data_cfg, 'visualize_stage')
    
    fake_batch = multi_loader.get_next_batch('fake')
    fake_images = fake_batch[1]
    fake_features = fake_batch[2]
    fake_z = fake_batch[0]
    fake_batches = [("fake", fake_z, fake_features, fake_images)]
    
    real_batches = []
    for key in multi_loader.keys():
        if key != 'fake':
            batch = multi_loader.get_next_batch(key)
            real_ims = batch[0]
            real_features = batch[1]
            real_batches.append( (key, real_features, real_ims) )
    
    
    ###### Models ########
    from models import load_torch_class
#     cifar_stylegan_net = load_torch_class('stylegan2-ada-cifar10', filename= '/cfarhomes/krusinga/storage/repositories/stylegan2-ada-pytorch/pretrained/cifar10.pkl').cuda()    
#     phi_model = pickle.load(open('./models/autoencoder/phi_model_exp_4.pkl','rb')) #note exp2
#     model = pickle.load(open('./models/autoencoder/ae_model_exp_4.pkl','rb'))
    
    cifar_stylegan_net = load_torch_class('stylegan2-ada-cifar10', filename= data_cfg.visualize_stage.stylegan_file).cuda()    
    model = pickle.load(open(data_cfg.visualize_stage.model_file,'rb'))  
    
    
    
    model.eval()
    cifar_stylegan_net.eval()
    
    with torch.no_grad():
        ##### Loop over fake batches #####
        for name, z_codes, fake_features, fake_ims in fake_batches:
            fake_encoded_norm, fake_encoded_code = model.encode(fake_ims.cuda(), features=fake_features.cuda())
            reconstructed_fake, _ = model.decode(fake_encoded_norm, fake_encoded_code)
            fake2real = torch.tanh(reconstructed_fake).detach().cpu()

            
            print(name, "original")
            view_tensor_images(fake_ims)
            print(name, "Reconstructed image through autoencoder")
            view_tensor_images(reconstructed_fake)
            print(name, "z_codes decoded via autoencoder")
            view_tensor_images(fake2real)
            
            # Store these somehow

        ##### Loop over real batches #####
        for name, real_features, real_ims in real_batches:
            # Stylegan beauracracy
            class_constant = torch.zeros(10, device='cuda')
            class_constant[class_constant_stylegan] = 1
            classes = class_constant.repeat(real_ims.size(0),1)
            

            real_encoded_norm, real_encoded_code = model.encode(real_ims.cuda(), features=real_features.cuda())

            if 'aug' in name:
                print('Scaling back to sphere')
                scale_factor = 512**0.5
            else:
                scale_factor = real_encoded_norm.unsqueeze(1) #network predicted norm

            reconstructed_real, _ = model.decode(real_encoded_norm, real_encoded_code)
            reconstructed_real = torch.tanh(reconstructed_real).detach().cpu()

            real2stylegan_w = cifar_stylegan_net.mapping(scale_factor*real_encoded_code, classes)
            real2stylegan = cifar_stylegan_net.synthesis(real2stylegan_w, noise_mode='const', force_fp32=True)
            real2stylegan = real2stylegan.detach().cpu()
            
            print(name, "original")
            view_tensor_images(real_ims)
            print(name, "Reconstructed image through autoencoder")
            view_tensor_images(reconstructed_real)
            print(name, "Encodings decoded via stylegan")
            view_tensor_images(real2stylegan)            



