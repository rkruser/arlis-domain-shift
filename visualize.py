import matplotlib.pyplot as plt
import torch
import numpy
import torchvision

def extract_range(dset, rn):
    allpts = []
    for i in rn:
        allpts.append(dset[i].image)
    allpts = torch.stack(allpts)
    return allpts

def visualize_image_grid(ims, nrow=4):
    sample_grid = torchvision.utils.make_grid(ims, nrow=nrow)

    samples_processed = sample_grid.detach().permute(1,2,0).cpu().numpy()
    samples_processed = (samples_processed+1)/2
    plt.imshow(samples_processed)
    plt.show()

def view_top_and_bottom(dataset, scores, num_each=64):
    sorted_score_indices = torch.argsort(scores)
        
    top_all = extract_range(dataset, sorted_score_indices[-num_each:]).flip(dims=(0,))
    print("Top all ims")
    visualize_image_grid(top_all, nrow=8)
    
    print("Bottom all ims")
    bottom_all = extract_range(dataset, sorted_score_indices[:num_each])
    visualize_image_grid(bottom_all, nrow=8)
    
"""    
    class_labels = torch.zeros(len(dataset), dtype=torch.int64)
    for i in range(len(dataset)):
        y = dataset[i].label
        class_labels[i] = y
    
    mapped_class_labels = class_labels[sorted_score_indices]
    
    unique_labels = class_labels.unique()
    class_index_sets = []
    for y in unique_labels:
        y_index = (mapped_class_labels == y)
        class_index_sets.append(y_index.nonzero().squeeze(1))    
    
    
    for i in range(len(unique_labels)):
        label = unique_labels[i]
        label_indices = class_index_sets[i]
        label_sorted_score_indices = sorted_score_indices[label_indices]
        print("=======================")
        print("Top label {0}".format(label))
        top_label = extract_range(dataset, label_sorted_score_indices[-num_each:]).flip(dims=(0,)) #Need to flip the top to put best first
        visualize_image_grid(top_label, nrow=8)
        print("Bottom label {0}".format(label))
        bottom_label = extract_range(dataset, label_sorted_score_indices[:num_each])
        visualize_image_grid(bottom_label, nrow=8)
"""



def visualize(data, dataset, opts):
    regressor_output = data.encodings.detach().cpu()
    
    on_manifold = regressor_output[:,0]
    off_manifold = regressor_output[:,1]

    print("On manifold histogram")
    plt.hist(on_manifold.numpy(), bins=100, density=True)
    plt.show()

    print("Off manifold histogram")
    plt.hist(off_manifold.numpy(), bins=100, density=True)
    plt.show()

    print("On manifold")
    view_top_and_bottom(dataset, on_manifold)
    print("Off manifold")
    view_top_and_bottom(dataset, off_manifold)

