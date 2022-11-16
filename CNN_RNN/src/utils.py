import os
import numpy as np
import torch
from typing import List
import time
import wandb


class Vocabulary:
    """
    Class to organize all vocabulary of the corpus and dictionaries to help reference the word,
     it's index and it's pretrained embedding.
     param corpus: (list of lists) containing all sentences in the corpus
     param vocab_size: (int) total size of the allowable vocabulary
     param pretrained_emb_matrix: (np.array) with embeddings for each word
    """

    def __init__(self, corpus, vocab_size, pretrained_emb_matrix):
        self.vocab_size = int(vocab_size)
        self.idx_to_str = {0: '<PAD>', 1: '<UNK>'}
        self.str_to_idx = {j: i for i, j in self.idx_to_str.items()}
        self._words_vocabulary(corpus)
        self.idx_to_emb = self._create_idx_to_emb(pretrained_emb_matrix)

    def __len__(self):
        return len(self.idx_to_str)

    def _words_vocabulary(self, sentence_list):
        """
        Counts the vocabulary and populates the "str_to_idx" and "idx_to_str" dictionaries
        """
        counts = {}
        idx = 2  # idx = 0 and 1 are reserved for <PAD> and <UNK>

        len_sent = []
        # calculate counts of words
        for sentence in sentence_list:
            len_sent.append(len(sentence))
            for word in sentence:
                if word not in counts.keys():
                    counts[word] = 1
                else:
                    counts[word] += 1

        # Filter based on the size of the allowable vocab
        if self.vocab_size <= len(counts):
            counts = dict(sorted(counts.items(), key=lambda x: -x[1])[:self.vocab_size - idx])
        else:
            self.vocab_size = len(counts)
            counts = dict(sorted(counts.items(), key=lambda x: -x[1])[:self.vocab_size - idx])

        # create vocab
        for word in counts.keys():
            self.str_to_idx[word] = idx
            self.idx_to_str[idx] = word
            idx += 1

    def _create_idx_to_emb(self, word_embeddings_mx):
        """
        creates a dictionary that maps the word's index to its pre-trained embedding "idx_to_emb_dict".
        If not pre-trained then the embeddings are generated randomly.
        """
        # Initialize a dictionary to populate the
        if word_embeddings_mx is not None:
            # Creat the word_dictionary like ('word', index in word embedding matrix)
            str_mx_idx = np.array(range(0, word_embeddings_mx.shape[0]))
            str_to_mx_idx = dict(zip(word_embeddings_mx[:, 0], str_mx_idx))
        else:
            str_to_mx_idx = {}

        # Initialize an empy dictionary and populate it with the word indexes
        idx_to_emb_dict = {}
        for word in self.str_to_idx.keys():
            try:  # If the word is in the pretrained embedding
                idx_to_emb_dict[self.str_to_idx[word]] = word_embeddings_mx[str_to_mx_idx[word], 1:]
            except:  # If the word is not in the pretrained embedding or '<PAD>' or '<UNK>'
                if word == '<PAD>':
                    idx_to_emb_dict[0] = np.zeros(100)
                elif word == '<UNK>':
                    idx_to_emb_dict[1] = np.random.rand(100)
                else:
                    idx_to_emb_dict[self.str_to_idx[word]] = np.random.rand(100)

        return idx_to_emb_dict


def read_files(pos_filepath: str, neg_filepath: str):
    """
    Reads in the sentence data with the corresponding tags
    :param pos_filepath: filepath associated with positive sentences
    :param neg_filepath: filepath associated with negative sentences
    :return: lists of lists with all sentences, list of tags (0 for negative and 1 for positive)
    """
    # Find all files within the filepath
    pos_filelist = os.listdir(pos_filepath)
    neg_filelist = os.listdir(neg_filepath)

    # Initialize the lists of sentences and the tags
    list_sentences = []
    tags = [1 for i in pos_filelist] + [0 for i in neg_filelist]

    # Read in the positive sentences
    for file in pos_filelist:
        sentence = np.loadtxt(fname=f"{pos_filepath}/{file}", delimiter=" ", dtype=str).tolist()
        list_sentences.append(sentence)

    # Read in the negative sentences
    for file in neg_filelist:
        sentence = np.loadtxt(fname=f"{neg_filepath}/{file}", delimiter=" ", dtype=str).tolist()
        list_sentences.append(sentence)

    return list_sentences, tags


def read_in_word_embedding(filepath):
    """
    Read in the pre-trained word embedding from the filepath
    :param filepath: (string) the filepath fto the pre-trained embeddings
    :return: (np.array) size = (words by emb length)
    """
    # Read in the word embeddings
    word_embeddings = np.loadtxt(fname=filepath, delimiter=" ", skiprows=1, dtype=str)
    return word_embeddings[:, :-1]


class DataSet(torch.utils.data.Dataset):
    """
    Pytorch data set class to initiate the loading train and test data for use
    param sentences: (lists of lists) with each internal list corresponding to a sentence
    param taggings: (list) the tags corresponding to each sentence
    param voc: (Vocabulary) vocab class from this utils file. Contains all dictionaries for word reference
    param max_len: (int) the maximum sentence size
    """

    def __init__(self, sentences: List, taggings: List, voc: Vocabulary, max_len: int):
        self.max_len = max_len
        self.sentences = np.array(self._trim_sentences(sentences))
        self.taggings = np.array(taggings)
        self.voc = voc

    def __len__(self):
        return len(self.taggings)

    def __getitem__(self, ind):
        # X = np.array([self.voc.str_to_idx[word] for word in self.sentences[ind]])
        X = [self.voc.str_to_idx[word] if word in self.voc.str_to_idx.keys() else 1
             for word in self.sentences[ind]]
        y = self.taggings[ind]
        return X, y

    def _trim_sentences(self, sentences):
        """Trims each sentence to the corrects size. If to small: adding padding, if too large cut"""
        new_sentence_list = []
        for sentence in sentences:
            if len(sentence) > self.max_len:  # Cut the sentence if it is too long
                new_sentence_list.append(sentence[:self.max_len])
            elif len(sentence) < self.max_len:  # Add padding to the sentence if too small
                zeros = ['<PAD>' for i in range(self.max_len - len(sentence))]
                new_sentence_list.append(sentence + zeros)
            else:  # if the sentence is equal to the max length
                new_sentence_list.append(sentence)
        return new_sentence_list

    @staticmethod
    def collate_fn(batch):
        """collate the X and y for training"""
        batch_x, batch_y = zip(*batch)
        return torch.LongTensor(batch_x), torch.LongTensor(batch_y)


def train_one_epoch(model, loss_fn, optimizer, train_loader, test_loader, epoch):
    """
    Trains the model for one epoch
    :param model: (class) the CNN or RNN class
    :param loss_fn: (nn.loss) loss class for pytorch
    :param optimizer: (nn.optimizer) optimize class from pytorch
    :param train_loader: (Loader) train loader
    :param test_loader: (Loader) test Loader
    :param epoch: (int) corresponding to the current epoch
    :return: (floats) average loss, average training accuracy, and testing accuracy per epoch
    """
    # Start training and initialize variables for iterations
    model.train()
    train_loss = 0.0
    test_loss = 0.0
    num_batches = len(train_loader)
    train_accuracy = []
    test_accuracy = []

    print(f"-- EPOCH {epoch} --")
    for (X, y_true) in train_loader:
        # Clear the gradients from previous iteration.
        optimizer.zero_grad()

        # Forward pass of the network. It will return the predictions.
        predictions = model.forward(X)

        # Calculate the loss of the predictions with respect to the actual labels.
        loss = loss_fn(predictions, y_true)
        model.eval()
        train_accuracy.append(find_accuracy(predictions, y_true))
        model.train()

        # Contact the backward pass and take step
        loss.backward()
        optimizer.step()

        # Adding the loss of current batch to the sum of loss of this epoch.
        train_loss = train_loss + loss.item()

    # Find the test accuracy and test loss
    model.eval()
    for (X, y_true) in test_loader:
        y_test_pred = model(X)
        loss = loss_fn(y_test_pred, y_true)
        test_loss = test_loss + loss.item()
        test_accuracy.append(find_accuracy(y_test_pred, y_true))

    # Return the loss and accuracy for the test and train
    avg_train_loss = train_loss / num_batches
    avg_test_loss = test_loss / num_batches

    return avg_train_loss, avg_test_loss, np.mean(train_accuracy), np.mean(test_accuracy)


def train_model(model, train_loader, test_loader,
                learning_rate: float = 1e-2,
                n_epochs: int = 10,
                use_wandb: str = 'N'):
    """
    Train the model prints of the metrics and saves all preformance as a csv
    :param model: (class) the CNN or RNN class
    :param train_loader: (Loader) train loader
    :param test_loader: (Loader) test Loader
    :param learning_rate: (float) the learning rate
    :param n_epochs: (int) the number of epochs
    :param use_wandb: (str) 'Y' or 'N' indicating the use of wandb
    """
    # Initiate the loss and optimization
    loss_fn = torch.nn.CrossEntropyLoss()  # or torch.nn.BCELoss()
    optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters())

    # Iterate over each epoch
    print(f"TRAINING: {model.__str__()}")
    start = time.time()
    for epoch in range(n_epochs):
        avg_train_loss, avg_test_loss, avg_train_acc, avg_test_acc \
            = train_one_epoch(model, loss_fn, optimizer, train_loader, test_loader, epoch)

        # Log the parameters to wandb
        if use_wandb == 'Y':
            wandb_log_dict = {"Train loss": avg_train_loss,
                              "Train accuracy": avg_train_acc,
                              "Test loss": avg_test_loss,
                              "Test accuracy": avg_test_acc,
                              'Epoch': epoch}
            wandb.log(wandb_log_dict)

        # Print out performance metrics
        print(f"Training Loss = {avg_train_loss}\nTesting Loss = {avg_test_loss}"
              f"\nTraining Accuracy = {avg_train_acc}\nTesting Accuracy = {avg_test_acc}")

    # Print the final metrics
    print(f"TRAINING COMPLETE, total time to train = {time.time() - start}")


def find_accuracy(predictions, y_true):
    """
    Finds the accuracy given predictions and labels
    :param predictions: (tensor) predictions
    :param y_true: (tensor) true labels
    :return: (float) accuracy
    """
    with torch.no_grad():
        counts = [torch.argmax(predictions, dim=1) == y_true]
        acc = sum(sum(counts))
        return acc / len(y_true)
