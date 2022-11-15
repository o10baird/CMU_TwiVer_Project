import argparse

import wandb
from torch.utils.data import DataLoader
from CNN import CNN
from LSTM import RNN
from utils import Vocabulary, DataSet, read_files, read_in_word_embedding, train_model


def main(model_type: str, pretrain_emb: str, voc_size: int, use_wandb: str):
    """
    1) Reading in the data and tags
    2) Determining if pretrained embeddings are used and read in
    3) Initializing the total vocabulary with the Vocabulary class
    4) Read data in to DataSet class to prepare for the DataLoader
    5) Read DataSet class into the DataLoaders
    6) Initiate CNN and RNN models
    7) Train Models
    :param pretrain_emb: (str) 'Y' indicated to use pre-trained embeddings, 'N' indicating do not use pre-trained embeddings
    :param voc_size: (int) the size of the permitted vocabulary
    """
    # Read in the data and tags
    train_sentences, train_tags = read_files("../data/train/positive", "../data/train/negative")
    test_sentences, test_tags = read_files("../data/test/positive", "../data/test/negative")

    # Read in the word embedding and the training data
    if pretrain_emb == 'Y':
        pretrained_embeddings = read_in_word_embedding("../data/all.review.vec.txt")
        pre_train_emb = True
    else:
        pretrained_embeddings = None
        pre_train_emb = False

    # Initialize the vocabulary and populate the vocab
    vocab = Vocabulary(corpus=train_sentences,
                       pretrained_emb_matrix=pretrained_embeddings,
                       vocab_size=voc_size)

    # load data into class
    max_sentence_len = 120
    train_data = DataSet(train_sentences, train_tags, vocab, max_sentence_len)
    test_data = DataSet(test_sentences, test_tags, vocab, max_sentence_len)

    # Load all the data into the pytorch data object using the inherited structure
    batch_size = 128
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, num_workers=2, collate_fn=train_data.collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                             shuffle=False, num_workers=2, collate_fn=test_data.collate_fn)

    # Call the correct model
    fully_connected_layer = 100
    if model_type == 'CNN':
        size_filter = 3
        n_filters = 100
        model = CNN(size_filter, n_filters, fully_connected_layer, vocab, max_sentence_len, pre_train_emb)
    elif model_type == 'RNN':
        model = RNN(fully_connected_layer, vocab, max_sentence_len, pre_train_emb)

    # Initiate training variables
    learning_rate = 1e-3
    n_epochs = 10

    # Start Wandb if used
    if use_wandb == 'Y':
        wandb.login()
        wandb_config = {
            "model_type": model.__str__(),
            "learning_rate": learning_rate,
            "epochs": n_epochs,
            "batch_size": batch_size}
        run = wandb.init(project="10641-HW3", config=wandb_config, reinit=True, name=f"{model_type}_pretrain{pretrain_emb}")
        wandb.watch(model, log="all")

    # Open wandb/run model/close wandb
    train_model(model, train_loader, test_loader, learning_rate, n_epochs,use_wandb)
    if use_wandb == 'Y':
        run.finish()


if __name__ == "__main__":
    # Select the arguments from the prompt
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['CNN', 'RNN'])  # pretrained embeddings
    parser.add_argument('--emb', choices=['Y', 'N'])  # pretrained embeddings
    parser.add_argument('--voc', type=int, default=10000)  # Vocab size
    parser.add_argument('--wandb', choices=['Y', 'N'])  # Use wandb or not
    args = parser.parse_args()

    main(args.model, args.emb, args.voc, args.wandb)
