from ae_models import *
from ae_data import *
from ae_method import view_tensor_images
from ae_combined import Feature_Encoder, Feature_Decoder
from utils import EasyDict as edict





class ExplicitBatchNorm(nn.Module):
    def __init__(self, input_shape=(512,), momentum=0.1, eps=1e-5, track_running_stats = True, device='cuda:0'):
        super().__init__()

        self.track_running_stats = track_running_stats
        self.momentum=momentum
        self.eps = eps
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

        self.running_mean = torch.zeros(input_shape, device=device)
        self.running_var = torch.ones(input_shape, device=device)

    def forward(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=True)

        if self.training and self.track_running_stats:
            self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*batch_mean.detach()
            self.running_var = (1-self.momentum)*self.running_var + self.momentum*batch_var.detach()

        if self.training or (not self.track_running_stats):
            sigma = torch.sqrt(batch_var+self.eps)
            mu = batch_mean
        else:
            sigma = torch.sqrt(self.running_var+self.eps)
            mu = self.running_mean

        x = self.gamma*((x - mu.unsqueeze(0))/sigma.unsqueeze(0)) + self.beta

        log_probs = (torch.log(self.gamma) - torch.log(sigma)).sum()
        
        return x, log_probs.repeat(x.size(0))


    # only invert in eval mode
    def invert(self, x):
        sigma = torch.sqrt(self.running_var+self.eps)
        x = sigma.unsqueeze(0)*((x-self.beta)/self.gamma) + self.running_mean.unsqueeze(0)

        log_probs = (torch.log(sigma) - torch.log(self.gamma)).sum()

        return x, log_probs.repeat(x.size(0))


class FlowNet(nn.Module):
    def __init__(self, num_blocks = 4, input_size=512, tan_coeff = 4, batchnorm_momentum=0.1, device='cuda:0', use_batchnorm=True, use_batchnorm_in_phi=True):
        super().__init__()

        assert(input_size%2 == 0)

        self.input_size = input_size
        self.num_blocks = num_blocks
        self.tan_coeff = tan_coeff
        self.device = device
    
        self.layer_constants = torch.eye(num_blocks*2).to(device)

        self.layer_selection_T = nn.Linear(num_blocks*2, input_size*2, bias=False)
        self.layer_selection_S = nn.Linear(num_blocks*2, input_size*2, bias=False)


        if use_batchnorm_in_phi:
            norm_layer = nn.BatchNorm1d
        else:
            norm_layer = nn.LayerNorm

        self.T_net = Phi(num_in = input_size//2, num_out=input_size//2, nblocks=4, hidden=input_size*2, norm_layer=norm_layer)
        self.S_net = Phi(num_in = input_size//2, num_out=input_size//2, nblocks=4, hidden=input_size*2, norm_layer=norm_layer)


        self.use_batchnorm = use_batchnorm

        if use_batchnorm:
            self.batch_norms = nn.ModuleList([ExplicitBatchNorm(input_shape=(input_size,), momentum=batchnorm_momentum) for i in range(num_blocks-1)])


    def run_block_forward(self, x, block_num):
#        print("*****************", block_num, "********************")


        block_const_1 = self.layer_constants[2*block_num]
        block_const_2 = self.layer_constants[2*block_num+1]
        
        T_const_1 = self.layer_selection_T(block_const_1).unsqueeze(0)
        S_const_1 = self.layer_selection_S(block_const_1).unsqueeze(0)
        T_const_2 = self.layer_selection_T(block_const_2).unsqueeze(0)
        S_const_2 = self.layer_selection_S(block_const_2).unsqueeze(0)

        x1, x2 = torch.split(x, [self.input_size//2, self.input_size//2], dim=1)
        
        tx1 = self.T_net.first_layer(x1) + T_const_1
        tx1 = self.T_net.main_layers(tx1)
        tx1 = self.T_net.last_layer(tx1)
        tx1 = self.tan_coeff*torch.tanh(tx1) # help numerically
        
        log_probs = tx1.sum(dim=1)

        sx1 = self.S_net.first_layer(x1) + S_const_1
        sx1 = self.S_net.main_layers(sx1)
        sx1 = self.S_net.last_layer(sx1)

        
        x2 = torch.exp(tx1)*x2 + sx1


#        print("********x2******\n", x2)

        tx2 = self.T_net.first_layer(x2) + T_const_2
        tx2 = self.T_net.main_layers(tx2)
        tx2 = self.T_net.last_layer(tx2)
        tx2 = self.tan_coeff*torch.tanh(tx2)

        log_probs += tx2.sum(dim=1)

        sx2 = self.S_net.first_layer(x2) + S_const_2
        sx2 = self.S_net.main_layers(sx2)
        sx2 = self.S_net.last_layer(sx2)
       
        x1 = torch.exp(tx2)*x1 + sx2        

#        print("**********x1*********\n", x1)

        return torch.cat([x1,x2], dim=1), log_probs


    def run_block_reverse(self, x, block_num):

        block_const_1 = self.layer_constants[2*block_num]
        block_const_2 = self.layer_constants[2*block_num+1]
        
        T_const_1 = self.layer_selection_T(block_const_1).unsqueeze(0)
        S_const_1 = self.layer_selection_S(block_const_1).unsqueeze(0)
        T_const_2 = self.layer_selection_T(block_const_2).unsqueeze(0)
        S_const_2 = self.layer_selection_S(block_const_2).unsqueeze(0)

        x1, x2 = torch.split(x, [self.input_size//2, self.input_size//2], dim=1)
        
        tx2 = self.T_net.first_layer(x2) + T_const_2
        tx2 = self.T_net.main_layers(tx2)
        tx2 = self.T_net.last_layer(tx2)
        tx2 = self.tan_coeff*torch.tanh(tx2)

        log_probs = -tx2.sum(dim=1)

        sx2 = self.S_net.first_layer(x2) + S_const_2
        sx2 = self.S_net.main_layers(sx2)
        sx2 = self.S_net.last_layer(sx2)
       
        x1 = torch.exp(-tx2)*(x1 - sx2)

        tx1 = self.T_net.first_layer(x1) + T_const_1
        tx1 = self.T_net.main_layers(tx1)
        tx1 = self.T_net.last_layer(tx1)
        tx1 = self.tan_coeff*torch.tanh(tx1)
        
        log_probs += -tx1.sum(dim=1)

        sx1 = self.S_net.first_layer(x1) + S_const_1
        sx1 = self.S_net.main_layers(sx1)
        sx1 = self.S_net.last_layer(sx1)
        
        x2 = torch.exp(-tx1)*(x2 - sx1)


        return torch.cat([x1,x2], dim=1), log_probs
       

        
    def forward(self, x):
        log_probs = torch.zeros(x.size(0), device=x.device)
        for num in range(self.num_blocks):
            x, lp = self.run_block_forward(x, num)
            log_probs += lp
       #     print(x)
            if self.use_batchnorm and (num < (self.num_blocks-1)):
                x, lp2 = self.batch_norms[num].forward(x)
                log_probs += lp2
            
        
        return x, log_probs
            
    def invert(self, y):
        log_probs = torch.zeros(y.size(0), device=y.device)
        for num in range(self.num_blocks-1,-1,-1):
            if num < (self.num_blocks-1):
                y, lp2 = self.batch_norms[num].invert(y)
                log_probs += lp2
            y, lp = self.run_block_reverse(y, num)
            log_probs += lp


        
        return y, log_probs
       


class AE_Flow(nn.Module):
    def __init__(self, lr=0.0001):
        super().__init__()
        lean = False
        very_lean = False
        all_linear = False
        add_linear = False
        use_layer_norm = False #use it in encoder/decoder
        use_linear_nets = False
        device = 'cuda:0'

        self.encoder = Encoder(32, ncolors=3, lean=lean, very_lean=very_lean, all_linear=all_linear, add_linear=add_linear, use_layer_norm = use_layer_norm).to(device)
        self.decoder = Decoder(32, ncolors=3, lean=lean, very_lean=very_lean, all_linear=all_linear, add_linear=add_linear, use_layer_norm=use_layer_norm).to(device)


        self.feature_encode = Feature_Encoder(linear=use_linear_nets).to(device)
        self.feature_decode = Feature_Decoder(linear=use_linear_nets).to(device)

        self.flow = FlowNet(input_size=512, num_blocks=4, tan_coeff=1, batchnorm_momentum=0.1).to(device)


        self.optim = torch.optim.Adam(self.parameters(), lr=lr)


    def forward(self, x, features):
        x = self.encoder(x)
        x = x.reshape(x.size(0), -1)
        
        x = self.feature_encode(x,features)
        
        z, logprobs = self.flow.forward(x)

        x, features = self.feature_decode(x)
        x = x.reshape(x.size(0), 32, 4, 4)
        x = self.decoder(x)

        return x, features, logprobs, z # reconstructed (x, features) and predicted z codes


    def z_sample(self, z):
        x, logprobs = self.flow.invert(z)
        logprobs = -logprobs #since inverted
        
        x, features = self.feature_decode(x)
        x = x.reshape(x.size(0), 32, 4, 4)
        x = self.decoder(x)
       
        return x, features, logprobs



def train_flow(model, dataloader, n_epochs, print_every=100):
    model.train()

    l2_lossfunc = torch.nn.MSELoss()

    for epoch in range(n_epochs):
        print("***************************epoch",epoch, "*************************")
        for i, batch in enumerate(dataloader):
#            x_real = batch['real'][0].cuda()
            x_fake = batch['fake'][1].cuda()
            z_fake = batch['fake'][0].cuda()

#            x_real_feats = batch['real'][-2].cuda()
            x_fake_feats = batch['fake'][-2].cuda()
           
#            x_recon_real, feat_recon_real, _, _ = model.forward(x_real, x_real_feats)
#           x_recon_real = torch.tanh(x_recon_real)
            x_recon_fake, feat_recon_fake, _, z_pred_fake = model.forward(x_fake, x_fake_feats)
            x_recon_fake = torch.tanh(x_recon_fake)
           
#            x_real_recon_loss = l2_lossfunc(x_recon_real, x_real)
            x_real_recon_loss = torch.tensor(0.0)

            x_fake_recon_loss = l2_lossfunc(x_recon_fake, x_fake)
#            feat_recon_real_loss = l2_lossfunc(x_real_feats, feat_recon_real)
            feat_recon_real_loss = torch.tensor(0.0)

            feat_recon_fake_loss = l2_lossfunc(x_fake_feats, feat_recon_fake)
            z_loss = l2_lossfunc(z_pred_fake, z_fake)
            
            total_loss = x_real_recon_loss + x_fake_recon_loss + feat_recon_real_loss + feat_recon_fake_loss + z_loss


            if (i>1) and (i%print_every == 0):
                loss_str = "x_real_recon: {0}, x_fake_recon: {1}, real_feat_recon: {2}, fake_feat_recon: {3}, z_loss: {4}".format(x_real_recon_loss.item(), x_fake_recon_loss.item(), feat_recon_real_loss.item(), feat_recon_fake_loss.item(), z_loss.item())

                print(i)
                print(loss_str)


    
            model.zero_grad()
            total_loss.backward()
            model.optim.step()












def visualize_model_flow(model_path, model_name_prefix, data_cfg, class_constant_stylegan=1):
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
            recon_fake, _, _, _ = model.forward(fake_ims.cuda(), features=fake_features.cuda())
            recon_fake = torch.tanh(recon_fake).detach().cpu()
            
            fake2real, _, _ = model.z_sample(z_codes.cuda())
            fake2real = torch.tanh(fake2real)
            
            print(name, "original")
            view_tensor_images(fake_ims)
            print(name, "Reconstructed image through autoencoder")
            view_tensor_images(recon_fake)
            print(name, "z_codes decoded via autoencoder")
            view_tensor_images(fake2real)
            
            # Store these somehow

        ##### Loop over real batches #####
        for name, real_features, real_ims in real_batches:
            # Stylegan beauracracy
            class_constant = torch.zeros(10, device='cuda')
            class_constant[class_constant_stylegan] = 1
            classes = class_constant.repeat(real_ims.size(0),1)
            

            recon_real, _, _, z_encoded = model.forward(real_ims.cuda(), features=real_features.cuda())
            reconstructed_real = torch.tanh(recon_real).detach().cpu()

            real2stylegan_w = cifar_stylegan_net.mapping(z_encoded, classes)
            real2stylegan = cifar_stylegan_net.synthesis(real2stylegan_w, noise_mode='const', force_fp32=True)
            real2stylegan = real2stylegan.detach().cpu()
            
            print(name, "original")
            view_tensor_images(real_ims)
            print(name, "Reconstructed image through autoencoder")
            view_tensor_images(recon_real)
            print(name, "Encodings decoded via stylegan")
            view_tensor_images(real2stylegan)            






def extract_probabilities_flow(model_path, model_name_prefix, data_cfg):
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
            features = batch[1]
            features = features.cuda()

            ims_recon, feats_recon, logprobs, z_predicted = model.forward(ims, features)
            ims_recon = torch.tanh(ims_recon)

            e2z_jacobian_probs.append(logprobs)           

            
            # Save the predicted norms
            z_norm = torch.norm(z_predicted,dim=1)
            z_norm = z_norm.detach().cpu().squeeze(1)
            z_norms.append(z_norm)

            # detach the 512-dim z and compute log priors
            z_predicted = z_predicted.detach()
            z_log_priors = log_priors(z_predicted)
            logpriors.append(z_log_priors)

            
            
            _, _, z2e_probs = model.z_sample(z_predicted)
            z2e_jacobian_probs.append(-z2e_probs)
            
            diffs = (features - feats_recon).detach().cpu()
            e_differences.append(diffs)
            e_norms.append(torch.norm(diffs,dim=1))
            
            
            total_z2e_probs.append(z_log_priors-z2e_probs)
            total_e2z_probs.append(z_log_priors+logprobs)
        
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









def test_flow():
    with torch.no_grad():
        net = FlowNet(input_size=512, num_blocks=4, tan_coeff=1, batchnorm_momentum=0.1)
        net.eval()
    
#        print(net.parameters())
#        net.eval()
#        print(net)
        
        x = torch.randn(30,512)
#        print(x)
        
        y, fprobs = net.forward(x)
#        print(y)

        xprime, invprobs = net.invert(y)
#        print(xprime)

        print(fprobs)
        print(invprobs)
    
        print(torch.norm(x-xprime,dim=1))

def test_batchnorm():
    with torch.no_grad():
        net = ExplicitBatchNorm(input_shape=(512,), momentum=1)
        for param in net.parameters():
            print(param)

#        print(net.parameters())
#        print(net.training)
#        net.eval()
#        print(net.training)
        x = torch.randn(30,512)
        #print(x)
        
        y, fprobs = net.forward(x)
        #print(y)

        xprime, invprobs = net.invert(y)
        #print(xprime)
    
        print(torch.norm(x-xprime,dim=1))
       

if __name__=='__main__':
    test_flow()
#    test_batchnorm()
