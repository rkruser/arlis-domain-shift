echo "Train ae"
python autoencoder_method_linear.py --mode train_ae
echo "Visualize ae"
python autoencoder_method_linear.py --mode visualize_ae
echo "Encode samples"
python autoencoder_method_linear.py --mode encode_samples
echo "Train phi"
python autoencoder_method_linear.py --mode train_phi
echo "Extract probs"
python autoencoder_method_linear.py --mode extract_probs
echo "visualize model"
python autoencoder_method_linear.py --mode visualize_model
echo "Plot probs"
python autoencoder_method_linear.py --mode plot_probs


















