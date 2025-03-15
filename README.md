# Image-Captioning
IMAGE CAPTIONING USING LSTMs and CNN ON Flickr8k Dataset
Image Captioning using CNN and LSTM

Overview

This repository contains an implementation of an image captioning model using Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM). The model generates textual descriptions for input images by extracting image features using a pre-trained CNN and feeding them into an LSTM-based language model.

Features

Uses a pre-trained CNN (DenseNet201) to extract image features.

Employs an LSTM-based decoder to generate captions.

Trained on the Flickr8k dataset.

Supports evaluation metrics like BLEU score for caption accuracy.

Implements modifications for performance improvement.

Requirements

Ensure you have the following dependencies installed:

pip install tensorflow keras numpy pandas matplotlib nltk pillow tqdm

Dataset

Download and prepare the dataset:

Images: Flickr8k Dataset

Captions: Provided in Flickr8k.token.txt

Place images in the data/images/ directory and captions in data/captions.txt.

Model Architecture

CNNs + LSTMs

The model consists of two main components:

Feature Extraction (CNN - DenseNet201): Extracts high-level image features.

Caption Generation (LSTM): Generates text captions based on extracted features.

Caption Text Preprocessing

Convert sentences into lowercase.

Remove special characters and numbers.

Remove extra spaces and single characters.

Add start (<start>) and end (<end>) tokens.

Tokenize and encode words into a one-hot representation.

Data Generation

Since training an image captioning model is resource-intensive, data is processed in batches. The image embeddings and their corresponding caption text embeddings are used as input for training. During inference, text embeddings are passed word by word to generate captions.

Training

Run the following command to preprocess the data and train the model:

python train.py

Training Details:

Optimizer: Adam

Loss Function: Categorical Crossentropy

Batch Size: 64

Epochs: 50 (early stopping enabled)

Training Results:

Epoch 1: Loss: 4.3540 | Val Loss: 3.6226

Epoch 5: Loss: 3.0592 | Val Loss: 3.0830

Epoch 10: Loss: 2.7585 | Val Loss: 3.0616

Final Epoch: Early stopping at Epoch 13 due to minimal improvement.

Inference

To generate captions for a new image, run:

python inference.py --image_path path/to/image.jpg

Evaluation

To evaluate the model using BLEU score:

python evaluate.py

Results

Sample output:

Input Image: 

Generated Caption: "A dog running through the water."

Observed issues: Some redundant captions and overuse of words like "blue shirt."

Possible Improvements:

Train on a larger dataset like Flickr30k or MSCOCO.

Implement attention mechanisms to improve accuracy.

Fine-tune hyperparameters for better generalization.

Model Enhancements

Modification: Image feature embeddings are added to LSTM outputs before being passed to fully connected layers.

Reference Paper: Show and Tell: A Neural Image Caption Generator

Future Work

Experiment with transformer-based models for better caption generation.

Leverage attention mechanisms for interpretability and accuracy.

Implement beam search decoding to generate more coherent captions.

Acknowledgments

Dataset: Flickr8k

Pre-trained CNN models: TensorFlow/Keras

Model architecture inspired by existing research papers.
