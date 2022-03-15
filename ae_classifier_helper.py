from ae_data import *
from utils import EasyDict as edict
from ae_combined import view_top_and_bottom

def extract_probabilities_classifier(model_path, model_name_prefix, data_cfg):
    print("In extract probs combined")
    multi_loader = get_dataloaders(data_cfg, 'prob_stage')   
    
    model = pickle.load(open(data_cfg.prob_stage.model_file,'rb'))
    model.eval()    

    stats = edict()
    #for name, dataloader in [("class_0", class_0_dataloader), ("class_1", class_1_dataloader),("fake_class_1",fake_class_1_dataloader)]:
        
    for name in multi_loader.keys():
        dataloader = multi_loader.loaders[name]
        
        print(name)
        classifier_outputs = []
        for i, batch in enumerate(dataloader):
            print("  {0} of {1}".format(i,len(dataloader)))
            ims = batch[0]
            ims = ims.cuda()
            features = batch[1]
            features = features.cuda()

            predicted_classes = model(ims, features=features).detach().cpu()
            classifier_outputs.append(predicted_classes)

        
        stats[name] = edict()
        #stats[name].e_codes = torch.cat(e_codes)
        stats[name].classifier_outputs = torch.cat(classifier_outputs)
        
    
    # 1. Encode the real data, detach encodings from graph
    # 2. Run encodings through e2z and z2e, get logprobs and log priors
    # 3. Plot (3 graphs: jacobian dets, priors, and combined)
    
    print(stats.keys())
    for key in stats:
        print(stats[key].keys())
        
    save_path = os.path.join(model_path, model_name_prefix+'_extracted.pkl')
    pickle.dump(stats, open(save_path,'wb'))    
        
    



def view_extracted_probabilities_classifier(model_path, model_name_prefix, data_cfg, aug_label="Real cars", aug_class=9):
    # note the off-manifold scores are norms here, not squared norms
    
    #data = pickle.load(open('./models/autoencoder/extracted_info_exp_4.pkl', 'rb'))
    data = pickle.load(open(data_cfg.plot_stage.prob_sample_file,'rb'))

    ######### View top and bottom of each ###########
    real_images = torch.load(data_cfg.visualize_stage.real)[1]['test']['images']
    fake_images = torch.load(data_cfg.visualize_stage.fake)[1]['test']['images']
    aug_images = torch.load(data_cfg.visualize_stage.augmented)[aug_class]['test']['images']

    real_e2z = data['real']['classifier_outputs']
    fake_e2z = data['fake']['classifier_outputs']
    aug_e2z = data['augmented']['classifier_outputs']

    view_top_and_bottom(real_e2z, real_images, 'Real cars, e2z')
    view_top_and_bottom(fake_e2z, fake_images, 'Fake cars, e2z')
    view_top_and_bottom(aug_e2z, aug_images, aug_label+', e2z')

    ###################

    
    plt.title("Classifier Outputs")
    plt.hist(data.real.classifier_outputs.numpy(), bins=50, density=True, alpha=0.3, label="Real cars")
    plt.hist(data.fake.classifier_outputs.numpy(), bins=50, density=True, alpha=0.3, label="Fake cars")    
    plt.hist(data.augmented.classifier_outputs.numpy(), bins=50, density=True, alpha=0.3, label=aug_label)
    plt.legend()
    plt.show()       
    

