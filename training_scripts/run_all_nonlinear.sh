echo "Train ae"
#python ae_method.py --mode train_ae --experiment_prefix nonlinear_ae --autoencoder_config_key nonlinear_ae
echo "Visualize ae"
#python ae_method.py --mode visualize_ae --experiment_prefix nonlinear_ae --autoencoder_config_key nonlinear_ae
echo "Encode samples"
#python ae_method.py --mode encode_samples --experiment_prefix nonlinear_ae --autoencoder_config_key nonlinear_ae
echo "Train phi"
python ae_method.py --mode train_phi --experiment_prefix nonlinear_ae --autoencoder_config_key nonlinear_ae
echo "Extract probs"
python ae_method.py --mode extract_probs --experiment_prefix nonlinear_ae --autoencoder_config_key nonlinear_ae
echo "visualize model"
python ae_method.py --mode visualize_model --experiment_prefix nonlinear_ae --autoencoder_config_key nonlinear_ae
echo "Plot probs"
python ae_method.py --mode plot_probs --experiment_prefix nonlinear_ae --autoencoder_config_key nonlinear_ae


















