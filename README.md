# Graph U-Net with Semi-Supervised Learning (GraphUnet-SS)

This project implements a Graph U-Net architecture for protein secondary structure prediction using semi-supervised learning. The system combines graph convolutional networks with U-Net style architecture for improved feature extraction from protein graphs.

## 📂 Project Structure
graph_unet_ss-main/
results/
deep_model.py # Main model execution file
scripts/
gends.py # Data generation and preparation
generate_n_window_dataset_for_fullpredict.py # Sliding window dataset
performance_metrics.py # Evaluation metrics
layers_gcn/ # Custom graph layers
graph_attention_cnn_layer.py
graph_cnn_layer.py
graph_convolutional_recurrent_layer.py
graph_ops.py
multi_graph_attention_cnn_layer.py
multi_graph_cnn_layer.py

text

## 🔧 Installation

### Requirements
- Python 3.7+
- TensorFlow >= 2.x
- Keras (TensorFlow integrated)
- scikit-optimize
- numpy

Install dependencies:
```bash
pip install numpy tensorflow keras scikit-optimize
🚀 Usage
Clone the repository:

bash
git clone https://github.com/ysngrmz/graph_unet_ss.git
cd graph_unet_ss
Run the main model:

bash
cd results
python deep_model.py
📜 Key Files
deep_model.py: Main training script with CNN, LSTM, and GraphCNN/GAT layers

gends.py: Data preparation and generation

generate_n_window_dataset_for_fullpredict.py: Creates windowed datasets

performance_metrics.py: Evaluation metrics calculation

layers_gcn/: Custom graph neural network layers

🧪 Features
Graph U-Net architecture for protein structure prediction

Semi-supervised learning approach

Bayesian hyperparameter optimization

Custom graph attention and convolutional layers

Sliding window dataset generation

📄 License
This project is currently unlicensed. Please contact the author for usage permissions. (Article information will be added)
