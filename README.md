# Generating Multimodal Representations for VQA using Graph Neural Networks

This repository aims at providing a GNN-based implementation to reason over both input modalities and improve performance on a VQA dataset. Here, the model receives an image `im` and a text-based question `t` and outputs the answer to the question `t`.

## Requirements

- `PyTorch`
- `PyTorch Geometric`
- `Numpy`
- `rsmlkit`

## Usage

- To pre-process and prepare the dataset for training, run `Dataset.py`
- To see the GNN-based model implementation, check `Model.py`
- `Match.py` is responsible for matching nodes locally via a graph neural network and then updating correspondence scores iteratively
- To see RNN-based Encoder-Decoder implementation and how it interacts with GNN model, check `Encoder_Decoder.py`
- `Parser.py` is responsible for instantiating the models as well as taking care of loading and saving of model checkpoints
- To train the whole model pipeline, run `Train.py`

## Results

The predicted answer to each question alongside its corresponsing image can be seen in the following attached output image:-
![alt text](https://github.com/fork123aniket/Graph-Neural-Network-based-Visual-Question-Answering/blob/main/Images/Result.png)
