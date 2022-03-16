import torch
import torchvision as tv
import numpy as np

from ae_models import *
from ae_data import *
from ae_combined import Feature_Encoder

from ae_flow import FlowNet
from models import load_torch_class


class Stylegan_Wrapper:
    def __init__(self, stylegan, sampled_class = None):
        self.stylegan = stylegan
        self.sampled_class = sampled_class

        if self.sampled_class is not None:
            self.class_constant = torch.zeros(10, device='cuda')
            self.class_constant[sampled_class] = 1


    def train(self):
        self.stylegan.train()

    def eval(self):
        self.stylegan.eval()

    def __call__(self, x):
        outputs = None
        if self.sampled_class is not None:
            class_inputs = self.class_constant.repeat(x.size(0),1)
            out_w = self.stylegan.mapping(x, class_inputs)
            outputs = self.stylegan.synthesis(out_w, noise_mode='const', force_fp32=True)
        else:
            out_w = self.stylegan.mapping(x)
            outputs = self.stylegan.synthesis(out_w, noise_mode='const', force_fp32=True)

        return outputs


def load_cifar_stylegan(filename, sampled_class = 1):
    cifar_stylegan_net = load_torch_class('stylegan2-ada-cifar10', filename = filename).cuda()

    return Stylegan_Wrapper(cifar_stylegan_net, sampled_class=sampled_class) 


def load_vgg16_pretrained():
    vgg16 = tv.models.vgg16(pretrained=True).cuda()
    return vgg16


# implement custom save function?
class Through_Model:
    def __init__(self, stylegan_file, lr=0.0001, use_features=False):
        self.encoder = Encoder(32, ncolors=3, lean=False, very_lean=False, all_linear=False, add_linear=True, use_layer_norm = True).cuda()
#        self.e2z = FlowNet(num_blocks=4, input_size=512, tan_coeff=1, use_batchnorm=False, use_batchnorm_in_phi=False).cuda()
        print("using phi e2z")
#        self.e2z = Phi(num_out=512, norm_layer=nn.LayerNorm).cuda()
#        self.adversary = Small_Classifier(linear=False, in_feats = 512).cuda()
        self.use_normalized = True
        self.e2z = Phi(num_out=513, norm_layer=nn.LayerNorm).cuda()
        self.adversary = Small_Classifier(linear=False, in_feats = 513).cuda()


        self.use_features = use_features
        if self.use_features:
            print("using features")
            self.feature_embedding = Feature_Encoder(linear=False).cuda()

        self.generator = load_cifar_stylegan(stylegan_file, sampled_class=1)
        self.features = load_vgg16_pretrained()

        self.generator.eval()
        self.features.eval()

        self.encoder_optim = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.e2z_optim = torch.optim.Adam(self.e2z.parameters(), lr=lr)
        if self.use_features:
            self.feature_embedding_optim = torch.optim.Adam(self.feature_embedding.parameters(), lr=lr)

        self.adversary_optim = torch.optim.Adam(self.adversary.parameters(), lr=lr)

    def train(self):
        self.encoder.train()
        self.e2z.train()
        self.adversary.train()

        if self.use_features:
            self.feature_embedding.train()

    def eval(self):
        self.encoder.eval()
        self.e2z.eval()
        self.adversary.eval()

        if self.use_features:
            self.feature_embedding.eval()


    def run(self, x, features=None, generate=True, get_features=False):
        e = self.encoder(x)
        if self.use_features or (generate and get_features):
            target_features = self.features(x).detach()
            if self.use_features:
                e = self.feature_embedding(e, target_features)
        else:
            target_features = None


        #z, _ = self.e2z(e)
        z = self.e2z(e)

        # normalize
        z_norms, z_normalized = torch.split(z, [1, 512], dim=1)
        z_norms = torch.nn.functional.elu(z_norms) + 1 #make everything positive
        z_normalized = z_normalized / torch.norm(z_normalized, dim=1).unsqueeze(1)
        z_final = z_norms*z_normalized



        if generate:
            gen_x = self.generator(z_final)
        else:
            gen_x = None

        if generate and get_features:
            gen_features = self.features(gen_x)
        else:
            gen_features = None
        

#        return gen_features, target_features, gen_x, z, e
        return gen_features, target_features, gen_x, z_norms, z_normalized, e



    def zero_grad(self):
        self.encoder_optim.zero_grad()
        self.e2z_optim.zero_grad()
        if self.use_features:
            self.feature_embedding_optim.zero_grad()

    def step_optim(self):
        self.encoder_optim.step()
        self.e2z_optim.step()
        if self.use_features:
            self.feature_embedding_optim.step()


    def run_adversary(self, real, fake):
        real_labels = torch.ones(real.size(0),device=real.device)
        fake_labels = torch.zeros(fake.size(0), device=fake.device)

        predicted_real = self.adversary(real)
        predicted_fake = self.adversary(fake)
        loss = 0.5*torch.nn.functional.binary_cross_entropy_with_logits(predicted_real, real_labels) + 0.5*torch.nn.functional.binary_cross_entropy_with_logits(predicted_fake, fake_labels)
        return loss
        
    def zero_adversary(self):
        self.adversary.zero_grad()

    def step_adversary(self):
        self.adversary_optim.step()



    

def norm_cosine_loss(x,y):
    x_norm = torch.norm(x,dim=1).unsqueeze(1)
    y_norm = torch.norm(y,dim=1).unsqueeze(1)

    x_scale = x / x_norm
    y_scale = y / y_norm

    cosines = 1.0 - (x_scale*y_scale).sum(dim=1)

    norm_loss = (x_norm - y_norm).square().mean()
    cosine_loss = cosines.mean()
    return norm_loss, cosine_loss


def train_through_model(model, dataloader, n_epochs, print_every=100, recon_lambda=1, adversary_lambda=1):
    print("Using lambda", recon_lambda)
    print("Using adversary lambda", adversary_lambda)
    model.train()

    l2_lossfunc = torch.nn.MSELoss()
    lossfunc = norm_cosine_loss

    phase = 0
    phase_0_count = 0
    for epoch in range(n_epochs):
        print("***************************epoch",epoch, "*************************")
        for i, batch in enumerate(dataloader):
            x_real = batch['real'][0].cuda()
            x_fake = batch['fake'][1].cuda()
            z_fake = batch['fake'][0].cuda()

            z_fake_norms = torch.norm(z_fake, dim=1)
            z_fake_normalized = z_fake / z_fake_norms.unsqueeze(1)

            if phase == 0:
                gen_feats_real, target_feats_real, gen_x_real, z_real_norms, z_real_normalized, _ = model.run(x_real, get_features=True)
                _, _, gen_x_fake, z_fake_pred_norms, z_fake_pred_normalized, _ = model.run(x_fake)
                
#                z_fake_norm_loss, z_fake_cosine_loss = lossfunc(z_fake_pred, z_fake)
                z_fake_norm_loss = l2_lossfunc(z_fake_pred_norms.squeeze(1), z_fake_norms)
                z_fake_cosine_loss = 1 - (z_fake_pred_normalized*z_fake_normalized).sum(dim=1).mean()


                pixel_loss = 0.5*l2_lossfunc(gen_x_fake, x_fake) + 0.5*l2_lossfunc(gen_x_real, x_real)
                real_feats_loss = l2_lossfunc(gen_feats_real, target_feats_real)

                adversary_loss = model.run_adversary(torch.cat([z_real_norms,z_real_normalized],dim=1),
                                                     torch.cat([z_fake_pred_norms,z_fake_pred_normalized],dim=1))

#                total_loss = z_fake_loss + recon_lambda*real_feats_loss - adversary_lambda*adversary_loss
                total_loss = z_fake_norm_loss + z_fake_cosine_loss + recon_lambda*real_feats_loss - adversary_lambda*adversary_loss + pixel_loss
       


                if (phase_0_count%print_every == 0):
#                    loss_str = "z_fake_loss = {0}, real_feats_loss = {1}, adversary_loss = {2}".format(z_fake_loss.item(), real_feats_loss.item(), adversary_loss.item())
                    loss_str = "z_fake_norm_loss = {0}, z_fake_cosine_loss = {3}, real_feats_loss = {1}, adversary_loss = {2}, pixel_loss = {4}".format(z_fake_norm_loss.item(), real_feats_loss.item(), adversary_loss.item(), z_fake_cosine_loss.item(), pixel_loss.item())


                    print(i)
                    print(loss_str)
        
                model.zero_grad()
                total_loss.backward()
                model.step_optim()

                phase_0_count += 1

            elif phase == 1:
                _, _, _, z_real_norms, z_real_normalized, _ = model.run(x_real, generate=False)
                _, _, _, z_fake_pred_norms,z_fake_pred_normalized, _ = model.run(x_fake, generate=False)

#                adv_loss = model.run_adversary(z_real.detach(), z_fake.detach())
                adv_loss = model.run_adversary(torch.cat([z_real_norms,z_real_normalized],dim=1),
                                                     torch.cat([z_fake_pred_norms,z_fake_pred_normalized],dim=1))

                model.zero_adversary()
                adv_loss.backward()
                model.step_adversary()
               
            phase = 1-phase                


# Ideas for ae through GAN:
#  - Much lower reconstruction lambda than z or adversary
#  - Zero out reconstruction loss until after a certain point (i.e. pretrain on fake only)
#  - Feature guidance still helps
#  - Is cosine + scaling more stable than L2 here?
# Experimental notes:
#  - I suspect that separating out the norm into a 513th number, then normalizing the output before returning, significantly stabilizes the model (must test)
#  - But also there may be batch size artifacts here (though since I'm using layernorm, that seems unlikely)
# Does adding pixel gradients do anything substantial?
# -----
# After some more experiments:
#  - Separating out that one index seems to help (it certainly fools the adversary, unlike before (make extra sure)), but also I think it's about batch size and the use of the generator model itself
#  - Need to see if cosines can go way down if you set the pixel/feature recon loss to 0 (and maybe adv loss)
#  - Other idea: still use the autoencoder, but take reconstruction losses from the GAN (seems to get best cosine losses this way?)
