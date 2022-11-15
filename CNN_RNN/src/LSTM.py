import numpy as np
import torch
from utils import Vocabulary


class RNN(torch.nn.Module):
    """
    Simple Recurrent Neural Network class for sentiment analysis
    param fully_connected_num: (int) fully connected layer num of connections
    param voc: (Vocabulary) class with all word dictionary references
    param max_sentence_len: (int) of max sentence length
    param pre_train_emb: (np.array) of the pre-trained embeddings
    """

    def __init__(self, fully_connected_num: int, voc: Vocabulary,
                 max_sentence_len: int, pre_train_emb: bool):
        super().__init__()

        # Attributes of the class
        self.max_sentence_len = max_sentence_len
        self.voc = voc
        self.fully_connected_num = fully_connected_num
        self.pre_trained_emb_array = self._create_emb_array()
        self.hidden_size = self.pre_trained_emb_array.size(1)

        # Layers of the class
        # 1) EMBEDDING LAYER
        if pre_train_emb:
            self.embedding = torch.nn.Embedding.from_pretrained(self.pre_trained_emb_array)
        else:
            self.embedding = torch.nn.Embedding(num_embeddings=self.voc.vocab_size,
                                                embedding_dim=self.fully_connected_num,
                                                padding_idx=0)

        # 2) LSTM Layer
        self.lstm = torch.nn.LSTM(input_size=self.max_sentence_len,
                                  hidden_size=self.hidden_size,
                                  batch_first=True)

        # 3) LINEAR LAYER
        self.linear = torch.nn.Linear(in_features=self.hidden_size,
                                      out_features=2)

        # self.model = nn.Sequential(self.embedding, self.conv, self.linear, self.activation)

    def __str__(self):
        return "RNN"

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
        output, (hidden, cell) = self.lstm(x)
        x = torch.squeeze(hidden)
        x = self.linear(x)
        return x
