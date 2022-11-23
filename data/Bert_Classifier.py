# !pip install transformers

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup as soup
import requests
import pandas as pd
import os
from nltk import tokenize
import nltk
nltk.download('punkt')
import time
import random

import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import re
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences


# make sure labels are not negetive

num_classes = 3
model_name = "bert-base-uncased" # model name from HuggingFace model card
X = list(df["text"].values) # data 
Y = list(df["label"].values) # labels

max_length = 512
tokenizer = BertTokenizerFast.from_pretrained(model_name)

train_texts, valid_texts, train_labels, valid_labels = train_test_split(X, Y, test_size=0.2)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

# convert our tokenized data into a torch Dataset
train_dataset = NewsDataset(train_encodings, train_labels)
valid_dataset = NewsDataset(valid_encodings, valid_labels)

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes).to("cuda")

from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  f1_micro = f1_score(labels, preds, average="micro")
  f1_macro = f1_score(labels, preds, average="macro")
  length = len(labels)
  return {
      'accuracy': acc,
      'f1-a': f1_micro,
      'f1-b': f1_macro,
      'len' : length,
  }
 
training_args = TrainingArguments(
    output_dir='./BERT_classification_results',          # output directory
    num_train_epochs=3,                                  # total number of training epochs
    per_device_train_batch_size=16,                      # batch size per device during training
    per_device_eval_batch_size=20,                       # batch size for evaluation
    warmup_steps=500,                                    # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                                   # strength of weight decay
    logging_dir='./BERT_classification_logs',            # directory for storing logs
    load_best_model_at_end=True,                         # load the best model when finished training (default metric is loss)
    metric_for_best_model = "accuracy",
    logging_steps=400,                                   # log & save weights each logging_steps
    save_steps=400,
    evaluation_strategy="steps",                         # evaluate each `logging_steps`
)

trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=valid_dataset,          # evaluation dataset
    compute_metrics=compute_metrics,     # the callback that computes metrics of interest
)

trainer.train()

trainer.evaluate()