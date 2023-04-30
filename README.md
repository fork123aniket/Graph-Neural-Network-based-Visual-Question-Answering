# Generating Multimodal Representations for VQA using Graph Neural Networks

This repository aims at providing a GNN-based implementation to reason over both input modalities and improve performance on a VQA dataset. Here, the model receives an image `im` and a text-based question `t` and outputs the answer to the question `t`. The following approach is being followed by the GNN model for the Visual Question Answering task:-

- Processing our input question into a graph G<sub>t</sub> and image into a graph G<sub>im</sub> using the Graph Parser.
- Passing the text graph G<sub>t</sub> and image graph G<sub>im</sub> into a graph neural network (GNN) to get the text and image node embeddings.
- Combining the embeddings using the Graph Matcher, which projects the text embeddings into the image embedding space and returns the combined multimodal representation of the input.
- Passing the joint representation through a sequence to sequence model to output the answer to the question.

## Requirements

- `PyTorch`
- `PyTorch Geometric`
- `Numpy`
- `rsmlkit`

## Usage

- This implementations trains the GNN model on the ***CLEVR*** dataset (a diagnostic dataset of 3D shapes that tests visual and linguistic reasoning), which can be downloaded from [***here***](https://cs.stanford.edu/people/jcjohns/clevr/).
- To pre-process and prepare the dataset for training, run `Dataset.py`
- To see the GNN-based model implementation, check `Model.py`
- `Match.py` is responsible for matching nodes locally via a graph neural network and then updating correspondence scores iteratively
- To see RNN-based Encoder-Decoder implementation and how it interacts with GNN model, check `Encoder_Decoder.py`
- `Parser.py` is responsible for instantiating the models as well as taking care of loading and saving of model checkpoints
- To train the whole model pipeline, run `Train.py`

## Results

The predicted answer to each question alongside its corresponsing image can be seen in the following attached output images:-

| Image | Question | GNN-Generated Answer|
| ----- |:--------:|:-------------------:|
| ![alt text](https://github.com/fork123aniket/Graph-Neural-Network-based-Visual-Question-Answering/blob/main/Images/Result.png) | There is a tiny matte thing that is left of the big gray cylinder and in front of the cyan metal cylinder; what color is it? | Purple |
| ![alt text](https://github.com/fork123aniket/Graph-Neural-Network-based-Visual-Question-Answering/blob/main/Images/Result.png) | There is a small blue block; are there any spheres to the left of it? | Yes |
