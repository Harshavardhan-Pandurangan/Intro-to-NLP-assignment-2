import torch
import torch.nn as nn
import conllu
import pandas as pd
import gensim.downloader as api
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import sys
import os

import fnn_trainer as fnn
import rnn_trainer as rnn
from fnn_trainer import FNNTrainer
from rnn_trainer import RNNTrainer

if __name__ == '__main__':
    # get all the command line arguments
    args = sys.argv

    if len(args) < 2:
        print("Please provide the name of the model to be loaded.")
        sys.exit(1)

    model_name = args[1]
    if model_name not in ['-f', '-r']:
        print("[-f for Feedforward Neural Network, -r for Recurrent Neural Network]")
        sys.exit(1)

    sentence = input()

    if model_name == '-f':
        # Feedforward Neural Network
        trainer = FNNTrainer()
        trainer.load_model('./fnn_model.pth')

        trainer.predict(sentence, 'glove-wiki-gigaword-200')

    elif model_name == '-r':
        pass

