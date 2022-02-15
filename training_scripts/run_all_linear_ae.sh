echo "Train ae"
python ae_method.py --mode train_ae
echo "Visualize ae"
python ae_method.py --mode visualize_ae
echo "Encode samples"
python ae_method.py --mode encode_samples
echo "Train phi"
python ae_method.py --mode train_phi
echo "Extract probs"
python ae_method.py --mode extract_probs
echo "visualize model"
python ae_method.py --mode visualize_model
echo "Plot probs"
python ae_method.py --mode plot_probs


















