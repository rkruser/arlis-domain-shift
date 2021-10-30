#
#   Read in a StyleGAN3 FFHQU file
#
from typing import List

import PIL
import numpy as np
import torch

from stylegan3 import dnnlib, legacy

model_file = 'models_stylegan3/stylegan3-r-ffhqu-256x256.pkl'
# model_file = '/home/clark/work/arlis/ryen/arlis-domain-shift/models_stylegan3/stylegan3-r-ffhqu-256x256.pkl'


def get_stylegan_ffhqu():
    print('Loading networks from "%s"...' % model_file)
    device = torch.device('cuda')
    with dnnlib.util.open_url(model_file) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
        return G


# Slightly modified from gen_images.py, since Tuple[float,float] didn't work right
def make_transform(translate: List[float], angle: float):
    m = np.eye(3)
    s = np.sin(angle / 360.0 * np.pi * 2)
    c = np.cos(angle / 360.0 * np.pi * 2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m


if __name__ == '__main__':
    # Read the network:
    G = get_stylegan_ffhqu()
    print(f"Num dimensions of latent:  {G.z_dim}")
    print("This is the network: ")
    print(f"{G}")

    # ------------------------
    # Generate using the network
    # ------------------------

    # Hard-coded values for demo.  See stylegan3/gen_images.py for actual setting
    device = torch.device('cuda')
    seed = 2
    translate = [0, 0]
    rotate = 0.
    label = torch.zeros([1, G.c_dim], device=device)
    truncation_psi = 1
    noise_mode = 'const'
    outdir = '/tmp'

    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

    # Construct an inverse rotation/translation matrix and pass to the generator.  The
    # generator expects this matrix as an inverse to avoid potentially failing numerical
    # operations in the network.
    if hasattr(G.synthesis, 'input'):
        m = make_transform(translate, rotate)
        m = np.linalg.inv(m)
        G.synthesis.input.transform.copy_(torch.from_numpy(m))

    #
    # WARNING!!!!:   The original line in gen_images is:
    #
    #     img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    #
    # But that did not work on my machine.  Apparently I have an older GPU at this point
    # so I have to force it to use FP32 (I have a:  GeForce GTX 1660 Ti)
    img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode, force_fp32=True)

    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')
