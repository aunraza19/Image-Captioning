##Image Captioning using CNN and LSTM

#Overview

This project implements an image captioning model using CNNs (DenseNet201) for feature extraction and LSTMs for caption generation. It is trained on the Flickr8k dataset and evaluates performance using BLEU scores.

#Features

Uses DenseNet201 for feature extraction.

LSTM-based decoder for caption generation.

Implements preprocessing, tokenization, and batch-wise data generation.

Supports evaluation and inference.

#Results

Sample Output: "A dog running through the water."

Issues: Repetitive phrases (e.g., “blue shirt”).

Improvements: Train on larger datasets (Flickr30k/MSCOCO) and implement attention mechanisms.

#Enhancements & Future Work

Improve accuracy using attention mechanisms and beam search decoding.

Experiment with transformers for better captioning.
