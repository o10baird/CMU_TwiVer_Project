import numpy as np
import torch
from utils import Vocabulary


class CNN(torch.nn.Module):
    """
    Simple Convolutional Neural Network class for sentiment analysis
    param filter_size: (int) size of the convolution silter
    param filter_num: (int) the numbers of filters to use
    param fully_connected_num: (int) fully connected layer num of connections
    param voc: (Vocabulary) class with all word dictionary references
    param max_sentence_len: (int) of max sentence length
    param pre_train_emb: (np.array) of the pre-trained embeddings
    """
    def __init__(self, filter_size: int, filter_num: int, fully_connected_num: int,
                 voc: Vocabulary, max_sentence_len: int, pre_train_emb: bool):
        super().__init__()

        # Attributes of the class
        self.filter_size = filter_size
        self.filter_num = filter_num
        self.voc = voc
        self.fully_connected_num = fully_connected_num
        self.pre_trained_emb_array = self._create_emb_array()
        self.conv_in_channel = self.pre_trained_emb_array.size(1)
        self.cov_out_channels = filter_num

        # Layers of the class
        # 1) EMBEDDING LAYER
        if pre_train_emb:
            self.embedding = torch.nn.Embedding.from_pretrained(self.pre_trained_emb_array)
        else:
            self.embedding = torch.nn.Embedding(num_embeddings=self.voc.vocab_size,
                                                embedding_dim=self.fully_connected_num,
                                                padding_idx=0)

        # 2) COVOLUTION LAYER
        self.conv = torch.nn.Conv1d(in_channels=self.conv_in_channel,
                                    out_channels=self.cov_out_channels,
                                    kernel_size=self.filter_size)

        # 3) ACTIVATION LAYER
        self.activation = torch.nn.ReLU()

        # 4) MAX POOL LAYER
        self.maxpool = torch.nn.MaxPool1d(kernel_size=max_sentence_len - (self.filter_size-1))

        # 5) LINEAR LAYER
        self.linear = torch.nn.Linear(in_features=self.conv_in_channel,
                                      out_features=2)

        # self.model = nn.Sequential(self.embedding, self.conv, self.linear, self.activation)

    def __str__(self):
        return "CNN"

    def _create_emb_array(self):
        """Create an embedding tensor for the max vocabulary"""
        indxs = self.voc.idx_to_emb.keys()
        emb_array = np.zeros((len(indxs), 100))
        for i in indxs:
            emb_array[i, :] = self.voc.idx_to_emb[i].astype(np.float)
        return torch.tensor(emb_array, dtype=torch.float)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.transpose(x, 1, 2)
        x = self.conv(x)
        x = self.activation(x)
        x = self.maxpool(x)
        x = torch.squeeze(x)
        x = self.linear(x)
        return x
