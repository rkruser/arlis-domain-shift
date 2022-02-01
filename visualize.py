import matplotlib.pyplot as plt
import torch
import numpy
import torchvision

def extract_range(dset, rn):
    if isinstance(dset, torch.Tensor):
        return dset[rn]
    else:
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

def view_top_and_bottom(dataset, scores, num_each=64, nrow=8):
    sorted_score_indices = torch.argsort(scores)
        
    top_all = extract_range(dataset, sorted_score_indices[-num_each:]).flip(dims=(0,))
    print("Top all ims")
    visualize_image_grid(top_all, nrow=nrow)
    
    print("Bottom all ims")
    bottom_all = extract_range(dataset, sorted_score_indices[:num_each])
    visualize_image_grid(bottom_all, nrow=nrow)
    



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
        visualize_image_grid(top_label, nrow=nrow)
        print("Bottom label {0}".format(label))
        bottom_label = extract_range(dataset, label_sorted_score_indices[:num_each])
        visualize_image_grid(bottom_label, nrow=nrow)
"""


"""
def view_top_and_bottom_both(dataset, on_scores, off_scores, num_each=64):
    on_sorted_inds = torch.argsort(on_scores)
    off_sorted_inds = torch.argsort(off_scores)
        
    top_all = extract_range(dataset, sorted_score_indices[-num_each:]).flip(dims=(0,))
    print("Top all ims")
    visualize_image_grid(top_all, nrow=8)
    
    print("Bottom all ims")
    bottom_all = extract_range(dataset, sorted_score_indices[:num_each])
    visualize_image_grid(bottom_all, nrow=8)
"""
    


def visualize(data, dataset, opts):
    regressor_output = data.encodings.detach().cpu()
    
    on_manifold = regressor_output[:,0]
    off_manifold = -regressor_output[:,1]
    combined = on_manifold + off_manifold

    print("On manifold histogram")
    plt.hist(on_manifold.numpy(), bins=100, density=True)
    plt.show()

    print("Off manifold histogram")
    plt.hist(-off_manifold.numpy(), bins=100, density=True)
    plt.show()

    print("On manifold")
    view_top_and_bottom(dataset, on_manifold, num_each=64, nrow=8)
    print("Off manifold")
    view_top_and_bottom(dataset, off_manifold, num_each=64, nrow=8)
    print("Combined")
    view_top_and_bottom(dataset, combined, num_each = 64, nrow = 8)


def visualize_compare(d1, d2, opts):
    r1 = d1.encodings.detach().cpu().numpy()
    r2 = d2.encodings.detach().cpu().numpy()

    label1_on = 'Cifar On-manifold'
    label2_on = 'Cifar 100 On-manifold'
    label1_off = 'Cifar off-manifold'
    label2_off = 'Cifar100 off-manifold'
    label1_combined = 'Cifar combined'
    label2_combined = 'Cifar100 combined'
    
    
    plt.hist(r1[:,0], bins=100, density=True, alpha=0.5, label=label1_on)
    plt.hist(r2[:,0], bins=100, density=True, alpha=0.5, label=label2_on)
    plt.title("Cifar10/Cifar100 On-manifold")
    plt.legend(loc='upper left')
    plt.show()

    plt.hist(-r1[:,1], bins=100, density=True, alpha=0.5, label=label1_off, range=[-7,5])
    plt.hist(-r2[:,1], bins=100, density=True, alpha=0.5, label=label2_off, range=[-7,5])
#    plt.hist(range=[-7,5])
    plt.title("Cifar10/Cifar100 Off-manifold")
    plt.legend(loc='upper left')
    plt.show()

    """
    plt.hist(r1[:,0]-r1[:,1], bins=100, density=True, alpha=0.5, label=label1_combined)
    plt.hist(r2[:,0]-r2[:,1], bins=100, density=True, alpha=0.5, label=label2_combined)
    plt.title("Cifar/MNIST Combined Scores")
    plt.legend(loc='upper left')
    plt.show()
    """
   

