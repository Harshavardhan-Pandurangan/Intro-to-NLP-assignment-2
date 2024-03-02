# NLP Assignment 1 - POS Tagging
## Harshavardhan P - 2021111003

<br>

## Running the codes

The models are run on cpu by default. To run on gpu, change the device to `cuda` in the `fnn_trainer.py` and `rnn_trainer.py` files.

### Training Models
```bash
python3 fnn_trainer.py
python3 rnn_trainer.py
```
Change the `fnn_trainer.py` and `rnn_trainer.py` to reflect required hyperparameters. The model classes for FNN and RNN have been changed to fixed activation functions (ReLU) for both and fixed hidden layer size (100) of RNN, due to repeated changes in hyperparameter tuning, and ease in implementation of `pos_tagger.py`.

### POS Tagging
```bash
python3 pos_tagger.py [-f -r]
```
Use `-f` for FNN and `-r` for RNN. Input the sentence to be POS tagged after running the command. The pretrained models are used for POS tagging. and are loaded from the `fnn_model.pth` and `rnn_model.pth` files, by load_model() method in `pos_tagger.py`.

### Experimentation
The experimentation is done in the `experiments.ipynb` file. The file contains the code for each component of the assignment, and can be changed easily to tune hyperparameters and run the models. The file also contains the code for the confusion matrix and accuracy calculation.
All metric calculations are done in the `experiments.ipynb` file, by changing the precision, recall, and f1-score functions to reflect the type of metric required (micro, macro, weighted), and the confusion matrix is generated manually by iterating through the predictions and actual labels.

### Graphs
The graphs asked in the assignment were generated in the `graphs.ipynb` file. The file contains the code for the graphs only asked in the assignment.