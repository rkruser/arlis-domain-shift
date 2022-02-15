echo "Train ae"
#python autoencoder_method_linear.py --mode train_ae --experiment_prefix nonlinear_ae --autoencoder_config_key nonlinear_ae
echo "Visualize ae"
#python autoencoder_method_linear.py --mode visualize_ae --experiment_prefix nonlinear_ae --autoencoder_config_key nonlinear_ae
echo "Encode samples"
#python autoencoder_method_linear.py --mode encode_samples --experiment_prefix nonlinear_ae --autoencoder_config_key nonlinear_ae
echo "Train phi"
python autoencoder_method_linear.py --mode train_phi --experiment_prefix nonlinear_ae --autoencoder_config_key nonlinear_ae
echo "Extract probs"
python autoencoder_method_linear.py --mode extract_probs --experiment_prefix nonlinear_ae --autoencoder_config_key nonlinear_ae
echo "visualize model"
python autoencoder_method_linear.py --mode visualize_model --experiment_prefix nonlinear_ae --autoencoder_config_key nonlinear_ae
echo "Plot probs"
python autoencoder_method_linear.py --mode plot_probs --experiment_prefix nonlinear_ae --autoencoder_config_key nonlinear_ae


















