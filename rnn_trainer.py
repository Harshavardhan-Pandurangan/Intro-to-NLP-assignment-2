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
import wandb

import warnings
warnings.filterwarnings('ignore')

device = ''
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
# device = torch.device("cpu")


class CoNLLUDataset(Dataset):
    def __init__(self, dataset, device):
        self.dataset = dataset
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_vector = torch.tensor(self.dataset[idx][0]).to(self.device)
        output_vector = torch.tensor(self.dataset[idx][1]).to(self.device)
        return input_vector, output_vector


# creating an RNN which takes n dim input and returns pos tag vector
class RNN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x


class RNNTrainer:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def setup_dataloaders(self, df_train, df_test, df_dev, embedding_type):
        train_data, word_vectors, pos_tags_one_hot = self.preprocess_train(df_train, embedding_type)
        dev_data = self.preprocess_dev_test(df_dev, word_vectors, pos_tags_one_hot)
        test_data = self.preprocess_dev_test(df_test, word_vectors, pos_tags_one_hot)

        train_conllu_dataset = CoNLLUDataset(train_data, self.device)
        dev_conllu_dataset = CoNLLUDataset(dev_data, self.device)
        test_conllu_dataset = CoNLLUDataset(test_data, self.device)

        train_dataloader = DataLoader(train_conllu_dataset, batch_size=64, shuffle=True)
        dev_dataloader = DataLoader(dev_conllu_dataset, batch_size=64, shuffle=True)
        test_dataloader = DataLoader(test_conllu_dataset, batch_size=64, shuffle=True)

        self.embed_dim = len(word_vectors['the'])
        self.output_dim = len(pos_tags_one_hot)
        self.pos_tags_one_hot = pos_tags_one_hot
        self.word_vectors = word_vectors

        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.test_dataloader = test_dataloader

        print('Dataloaders Created')

    def create_model(self, hidden_dim):
        self.model = RNN(self.embed_dim, hidden_dim, self.output_dim).to(self.device)

        print('Model Created')

    def setup_cr_op(self, criterion, optimizer):
        if criterion == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif criterion == 'bce':
            self.criterion = nn.BCELoss()
        else:
            print('Invalid Criterion')

        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        elif optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.02)
        else:
            print('Invalid Optimizer')

        print('Criterion and Optimizer Setup')

    def preprocess_train(self, df, embedding_type='glove-wiki-gigaword-100'):
        vocab = set(df['word'])
        pos_tags = set(df['pos'])
        word_vectors_all = api.load(embedding_type)

        word_vectors = {}
        for word in vocab:
            if word in word_vectors_all:
                word_vectors[word] = word_vectors_all[word]
            else:
                word_vectors[word] = np.zeros(len(word_vectors_all['the']))

            # one hot encode the pos tags
        pos_tags_one_hot = {}
        for i, tag in enumerate(pos_tags):
            one_hot = np.zeros(len(pos_tags) + 1)
            one_hot[i] = 1
            pos_tags_one_hot[tag] = one_hot
        pos_tags_one_hot[''] = np.zeros(len(pos_tags) + 1)
        pos_tags_one_hot[''][-1] = 1

        # convert the df to list
        data = df.values.tolist()
        dataset = [[word_vectors[data[i][0]], pos_tags_one_hot[data[i][1]]] for i in range(len(data))]

        return dataset, word_vectors, pos_tags_one_hot

    def preprocess_dev_test(self, df, word_vectors, pos_tags_one_hot):
        data = df.values.tolist()

        dataset = []
        for i in range(len(data)):
            word = data[i][0]
            if word in word_vectors:
                word_vector = word_vectors[word]
            else:
                word_vector = np.zeros(len(word_vectors['the']))
            if data[i][1] in pos_tags_one_hot:
                pos_vector = pos_tags_one_hot[data[i][1]]
            else:
                pos_vector = pos_tags_one_hot['']
            dataset.append([word_vector, pos_vector])

        return dataset

    def train(self, epochs, train_dataloader):
        print()
        self.model.train()
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            running_loss = 0.0
            for i, data in enumerate(tqdm.tqdm(train_dataloader)):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs.float())
                loss = self.criterion(outputs, labels.float())
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f'Loss: {epoch + 1}/{epochs}: {running_loss / len(train_dataloader)}')

    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for i, data in enumerate(tqdm.tqdm(dataloader)):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs.float())
                loss = self.criterion(outputs, labels.float())
                running_loss += loss.item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.cpu().numpy())

        print(f'Loss: {running_loss / len(dataloader)}')
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_labels = np.argmax(all_labels, axis=1)
        all_preds = np.argmax(all_preds, axis=1)
        print(f'Accuracy: {accuracy_score(all_labels, all_preds)}')
        print(f'Precision: {precision_score(all_labels, all_preds, average="weighted")}')
        print(f'Recall: {recall_score(all_labels, all_preds, average="weighted")}')
        print(f'F1: {f1_score(all_labels, all_preds, average="weighted")}')

        return {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average="weighted"),
            'recall': recall_score(all_labels, all_preds, average="weighted"),
            'f1': f1_score(all_labels, all_preds, average="weighted")
        }

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print('Model Saved')

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        print('Model Loaded')


if __name__ == '__main__':
    # import the data files
    dataset_path_train = 'ud-treebanks-v2.13/UD_English-Atis/en_atis-ud-train.conllu'
    dataset_path_dev = 'ud-treebanks-v2.13/UD_English-Atis/en_atis-ud-dev.conllu'
    dataset_path_test = 'ud-treebanks-v2.13/UD_English-Atis/en_atis-ud-test.conllu'

    dataset_train = conllu.parse_incr(open(dataset_path_train))
    dataset_dev = conllu.parse_incr(open(dataset_path_dev))
    dataset_test = conllu.parse_incr(open(dataset_path_test))

    print('Data Loaded')

    # create a dataframe from the data
    def create_dataframe(dataset):
        data = []
        for tokenlist in dataset:
            for token in tokenlist:
                # data.append([token['form'], token['upostag']])
                data.append([token['form'], token['upostag']])
        # return pd.DataFrame(data, columns=['', 'word', 'pos'])
        return pd.DataFrame(data, columns=['word', 'pos'])

    df_train = create_dataframe(dataset_train)
    df_dev = create_dataframe(dataset_dev)
    df_test = create_dataframe(dataset_test)

    print('Dataframes Created')

    rnn_trainer = RNNTrainer()
    rnn_trainer.setup_dataloaders(df_train, df_test, df_dev, 'glove-wiki-gigaword-200')
    rnn_trainer.create_model(100)
    rnn_trainer.setup_cr_op('bce', 'adam')
    rnn_trainer.train(20, rnn_trainer.train_dataloader)
    results_test = rnn_trainer.evaluate(rnn_trainer.test_dataloader)
    results_dev = rnn_trainer.evaluate(rnn_trainer.dev_dataloader)
    rnn_trainer.save_model('rnn_model.pth')

    # with open('results.csv', 'w') as file:
    #     # write the parameters used, along with the results of the test and dev sets, in a csv file
    #     file.write('hidden_dim,embed_dim,prev_n,succ_n,criterion,optimizer,epochs,accuracy_test,precision_test,recall_test,f1_test,accuracy_dev,precision_dev,recall_dev,f1_dev\n')
    #     file.write(str(100) + ',' + str(100) + ',' + str(0) + ',' + str(0) + ',' + 'bce' + ',' + 'adam' + ',' + str(10) + ',' + str(results_test['accuracy']) + ',' + str(results_test['precision']) + ',' + str(results_test['recall']) + ',' + str(results_test['f1']) + ',' + str(results_dev['accuracy']) + ',' + str(results_dev['precision']) + ',' + str(results_dev['recall']) + ',' + str(results_dev['f1']))
    #     file.write('\n')
    # print('Results Saved')

    # # connecting to wandb
    # wandb.init(project='RNN_POS_Tagger', entity='rockingharsha71')
    # for embedding in ['glove-wiki-gigaword-100', 'glove-wiki-gigaword-200', 'word2vec-google-news-300']:
    #     for hidden_dim in [50, 100, 200, 300]:
    #         for criterion in ['bce', 'cross_entropy']:
    #             print(f'Running for {embedding}, {hidden_dim}, {criterion}')

    #             rnn_trainer = RNNTrainer()
    #             rnn_trainer.setup_dataloaders(df_train, df_test, df_dev, embedding)
    #             rnn_trainer.create_model(hidden_dim)
    #             rnn_trainer.setup_cr_op(criterion, 'adam')
    #             rnn_trainer.train(10, rnn_trainer.train_dataloader)
    #             results_test = rnn_trainer.evaluate(rnn_trainer.test_dataloader)
    #             results_dev = rnn_trainer.evaluate(rnn_trainer.dev_dataloader)

    #             wandb.init(project='RNN_POS_Tagger', entity='rockingharsha71', reinit=True)
    #             wandb.log({
    #                 'embedding': embedding,
    #                 'hidden_dim': hidden_dim,
    #                 'criterion': criterion,
    #                 'accuracy_test': results_test['accuracy'],
    #                 'precision_test': results_test['precision'],
    #                 'recall_test': results_test['recall'],
    #                 'f1_test': results_test['f1'],
    #                 'accuracy_dev': results_dev['accuracy'],
    #                 'precision_dev': results_dev['precision'],
    #                 'recall_dev': results_dev['recall'],
    #                 'f1_dev': results_dev['f1']
    #             })

    #             print('Results Logged to Wandb')