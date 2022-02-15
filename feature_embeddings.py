import torch
import torchvision as tv
import numpy as np



# Take images of any size loaded into [-1,1] range
# Obtain vgg features of these images
def obtain_vgg_features(dataloader):
    vgg16 = tv.models.vgg16(pretrained=True).cuda()
    resize_func = tv.transforms.Resize(224)
    normalize_func = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    
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
            batch = batch.cuda()

            output = vgg16(batch)
            features.append(output.detach().cpu())
            
    return torch.cat(features)