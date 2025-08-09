# Graph U-Net with Semi-Supervised Learning (GraphUnet-SS)

This repository implements a **Graph U-Net** architecture for **protein secondary structure prediction** using a semi-supervised learning approach. The system leverages **graph convolutional networks (GCNs)** combined with a U-Net-style architecture to extract features from protein graphs, incorporating custom graph layers, Bayesian hyperparameter optimization, and sliding window dataset generation.

## 📚 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Key Files](#key-files)
- [Custom Graph Layers](#custom-graph-layers)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 📖 Overview

The **GraphUnet-SS** project is designed for protein secondary structure prediction, a critical task in bioinformatics. It employs a Graph U-Net architecture, which combines the strengths of graph convolutional networks and U-Net's encoder-decoder structure. The semi-supervised learning approach allows the model to leverage both labeled.Paths and unlabeled data, improving performance on sparse datasets. The project includes custom graph layers, data preprocessing scripts, and evaluation metrics, with support for Bayesian hyperparameter optimization.

## ✨ Features

- **Graph U-Net Architecture**: Combines graph convolutional networks with a U-Net-style encoder-decoder for robust feature extraction.
- **Semi-Supervised Learning**: Utilizes both labeled and unlabeled protein data to enhance model performance.
- **Custom Graph Layers**: Includes specialized layers like Graph Attention CNN and Graph Convolutional Recurrent layers.
- **Sliding Window Dataset Generation**: Supports windowed data preparation for protein sequences.
- **Bayesian Hyperparameter Optimization**: Uses `scikit-optimize` for efficient hyperparameter tuning.
- **Comprehensive Evaluation**: Includes performance metrics for model evaluation.

## 📂 Project Structure

The project is organized as follows:

graph_unet_ss/ ├── results/ │ └── deep_model.py # Main script for model training and execution ├── scripts/ │ ├── gends.py # Data generation and preprocessing │ ├── generate_n_window_dataset_for_fullpredict.py # Sliding window dataset creation │ └── performance_metrics.py # Evaluation metrics calculation ├── layers_gcn/ │ ├── graph_attention_cnn_layer.py # Graph Attention CNN layer implementation │ ├── graph_cnn_layer.py # Graph CNN layer implementation │ ├── graph_convolutional_recurrent_layer.py # Graph Convolutional Recurrent layer │ ├── graph_ops.py # Graph operations utilities │ ├── multi_graph_attention_cnn_layer.py # Multi-Graph Attention CNN layer │ └── multi_graph_cnn_layer.py # Multi-Graph CNN layer ├── README.md # This file


## 📋 Requirements

To run the project, ensure you have the following dependencies installed:

- **Python**: 3.7 or higher
- **TensorFlow**: 2.x or higher (includes Keras)
- **NumPy**: For numerical computations
- **scikit-optimize**: For Bayesian hyperparameter optimization

Optional (depending on specific requirements):
- Additional bioinformatics libraries (e.g., Biopython) may be required for protein data handling.

## 🔧 Installation

Follow these steps to set up the project:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ysngrmz/graph_unet_ss.git
   cd graph_unet_ss





Create a Virtual Environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate



Install Dependencies:

pip install numpy tensorflow scikit-optimize



Verify Installation: Ensure all dependencies are installed correctly by running:

python -c "import tensorflow, numpy, skopt; print('Dependencies installed successfully!')"

🚀 Usage

To train and evaluate the Graph U-Net model, follow these steps:





Prepare the Data:





Use scripts/gends.py to preprocess and generate protein graph datasets.



Run scripts/generate_n_window_dataset_for_fullpredict.py to create sliding window datasets for prediction.

cd scripts
python gends.py
python generate_n_window_dataset_for_fullpredict.py



Train the Model:





Navigate to the results directory and execute the main training script.

cd results
python deep_model.py



Evaluate Performance:





Use scripts/performance_metrics.py to compute evaluation metrics after training.

cd scripts
python performance_metrics.py

Note: Ensure that the input data (protein sequences or graphs) is properly formatted and placed in the appropriate directory as specified in gends.py or related scripts.

📜 Key Files





deep_model.py: The main script that orchestrates model training and inference. It integrates CNN, LSTM, and custom graph layers (from layers_gcn/) for protein secondary structure prediction.



gends.py: Handles data preprocessing, including loading protein sequences and constructing graph representations.



generate_n_window_dataset_for_fullpredict.py: Generates sliding window datasets for full sequence prediction.



performance_metrics.py: Calculates metrics such as accuracy, precision, recall, and F1-score for model evaluation.



layers_gcn/: Directory containing custom graph neural network layers (see below for details).

🧠 Custom Graph Layers

The layers_gcn/ directory contains specialized layers for graph-based processing:





graph_attention_cnn_layer.py: Implements a Graph Attention CNN layer, combining attention mechanisms with convolutional operations.



graph_cnn_layer.py: Standard Graph CNN layer for feature extraction on graph-structured data.



graph_convolutional_recurrent_layer.py: Combines graph convolutions with recurrent neural networks for temporal graph processing.



graph_ops.py: Utility functions for graph operations, such as adjacency matrix manipulation.



multi_graph_attention_cnn_layer.py: Multi-head version of the Graph Attention CNN layer.



multi_graph_cnn_layer.py: Multi-head Graph CNN layer for enhanced feature extraction.

These layers are designed to work seamlessly with TensorFlow and Keras, enabling flexible integration into the Graph U-Net architecture.

🤝 Contributing

Contributions are welcome! To contribute:





Fork the repository.



Create a new branch (git checkout -b feature/your-feature).



Make your changes and commit (git commit -m "Add your feature").



Push to the branch (git push origin feature/your-feature).



Open a pull request.

Please ensure your code follows the project's coding style and includes appropriate documentation.

📄 License

This project is currently unlicensed. For usage permissions, please contact the author (details to be added). Future updates may include a formal license and associated article information.

📬 Contact

For questions or support, please contact the project author through GitHub issues or the contact information provided in future updates.



Last updated: August 9, 2025

