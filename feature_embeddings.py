import torch
import torchvision as tv
import numpy as np
import pickle





def apply_vgg(vggnet, batch, device='cuda:1'):
    resize_func = tv.transforms.Resize(224)
    normalize_func = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    

    with torch.no_grad():
        batch = resize_func(batch)
        
        batch = ((batch+1)/2).clamp(0,1)
        batch = normalize_func(batch)
        batch = batch.to(device)

        output = vggnet(batch)
            
    return output

# Take images of any size loaded into [-1,1] range
# Obtain vgg features of these images
def obtain_vgg_features(dataloader, data_batch = None, device='cuda:0'):
    vgg16 = tv.models.vgg16(pretrained=True).to(device)
    resize_func = tv.transforms.Resize(224)
    normalize_func = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    

    if data_batch is not None:
        dataloader = torch.utils.data.DataLoader(data_batch, batch_size = min(64,data_batch.size(0)), shuffle=False)

    print("obtaining features")
    features = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i%10 == 0:
                print(i)
            if isinstance(batch, tuple):
                batch = batch[0]
                
            # assume batch is in [-1, 1]
            batch = resize_func(batch)
            
            batch = ((batch+1)/2).clamp(0,1)
            batch = normalize_func(batch)
            batch = batch.to(device)

            output = vgg16(batch)
            features.append(output.detach().cpu())
            
    return torch.cat(features)





def mean_diffs(x, mean=None):
    if mean is None:
        mean = x.mean(dim=0)

    diffs = x - mean
    normdiffs = torch.norm(diffs, dim=1)
    mean_normdiff = normdiffs.mean()
    stdev_normdiff = normdiffs.std()
    
    return mean_normdiff, stdev_normdiff

def vgg_differences():
    cifar = torch.load('./models/ae2/cifar_sorted.pth')

    key = 'encodings_vgg16'
#     c0 = cifar[0]['train'][key]
#     c1 = cifar[1]['train'][key]
#     c2 = cifar[2]['train'][key]
#     c3 = cifar[3]['train'][key]
#     c4 = cifar[4]['train'][key]
#     c5 = cifar[5]['train'][key]
#     c6 = cifar[6]['train'][key]
#     c7 = cifar[7]['train'][key]
#     c8 = cifar[8]['train'][key]
#     c9 = cifar[9]['train'][key]
    
    internal = []
    pairwise = []
    mean_internal = 0
    mean_pairwise = 0
    for i in range(10):
        ci = cifar[i]['train'][key]
        ci_mean, ci_std = mean_diffs(ci)
        internal.append( (i, ci_mean, ci_std) )
        
        mean_internal += ci_mean
        
        for j in range(i+1,10):
            cj = cifar[j]['train'][key]
            cij_mean, cij_std = mean_diffs(cj, mean=ci.mean(dim=0))
            pairwise.append( (i, j, cij_mean.item(), cij_std.item()) )
            mean_pairwise += cij_mean
            
    print("internal", internal)
    print("pairwise", pairwise)
    
    print('mean internal', mean_internal/10)
    print('mean between', mean_pairwise/45)


if __name__ == '__main__':
    vgg_differences()













