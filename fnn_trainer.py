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
import nltk

import warnings
warnings.filterwarnings("ignore")

# dataset class for the CoNLL-U dataset
class CoNLLUDataset(Dataset):
    def __init__(self, data, device=torch.device('cpu')):
        self.dataset = data
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # input_tensor = torch.tensor(self.dataset[idx][0], dtype=torch.float32, device=device)
        input_tensor = self.dataset[idx][0].to(self.device)
        # target_tensor = torch.tensor(self.dataset[idx][1], dtype=torch.float32, device=device)
        target_tensor = self.dataset[idx][1].to(self.device)
        return input_tensor, target_tensor


# creating an FNN which takes n dim input and returns pos tag vector
class FNN(nn.Module):
    def __init__(self, embed_dim, prev_n, succ_n, hidden_params, output_dim):
        super(FNN, self).__init__()
        # for each element in hidden_params, we will create a linear layer
        hidden_layers = []
        hidden_layers.append(nn.Linear(embed_dim * (prev_n + 1 + succ_n), hidden_params[0]))
        hidden_layers.append(nn.ReLU())
        for i in range(1, len(hidden_params)):
            hidden_layers.append(nn.Linear(hidden_params[i-1], hidden_params[i]))
            hidden_layers.append(nn.ReLU())
        # softmax layer for output
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(hidden_params[-1], output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x


class FNNTrainer:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def setup_dataloaders(self, df_train, df_test, df_dev, prev_n, succ_n, embedding_type):
        train_data, word_vectors, pos_tags_one_hot = self.preprocess_train(df_train, prev_n, succ_n, embedding_type)
        dev_data = self.preprocess_dev_test(df_dev, word_vectors, pos_tags_one_hot, prev_n, succ_n)
        test_data = self.preprocess_dev_test(df_test, word_vectors, pos_tags_one_hot, prev_n, succ_n)

        train_conllu_dataset = CoNLLUDataset(train_data, self.device)
        dev_conllu_dataset = CoNLLUDataset(dev_data, self.device)
        test_conllu_dataset = CoNLLUDataset(test_data, self.device)

        torch.manual_seed(0)

        train_dataloader = DataLoader(train_conllu_dataset, batch_size=64, shuffle=True)
        dev_dataloader = DataLoader(dev_conllu_dataset, batch_size=64, shuffle=True)
        test_dataloader = DataLoader(test_conllu_dataset, batch_size=64, shuffle=True)

        self.embed_dim = len(word_vectors['the'])
        self.prev_n = prev_n
        self.succ_n = succ_n
        self.output_dim = len(pos_tags_one_hot)
        self.pos_tags_one_hot = pos_tags_one_hot
        self.word_vectors = word_vectors

        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.test_dataloader = test_dataloader

        print('Dataloaders Created')

    def create_model(self, hidden_params, embed_dim=None, prev_n=None, succ_n=None, output_dim=None):
        self.hidden_params = hidden_params
        self.model =  FNN(self.embed_dim, self.prev_n, self.succ_n, hidden_params, self.output_dim)

    def setup_cr_op(self, criterion, optimizer):
        criterion_ = None
        optimizer_ = None
        if criterion == 'cross_entropy':
            criterion_ = nn.CrossEntropyLoss()
            self.criterion_type = 'cross_entropy'
        elif criterion == 'bce':
            criterion_ = nn.BCELoss()
            self.criterion_type = 'bce'
        else:
            print('Invalid criterion')

        if optimizer == 'adam':
            optimizer_ = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)
            self.optimizer_type = 'adam'
        elif optimizer == 'sgd':
            optimizer_ = torch.optim.SGD(self.model.parameters(), lr=0.001)
            self.optimizer_type = 'sgd'
        else:
            print('Invalid optimizer')

        self.criterion = criterion_
        self.optimizer = optimizer_

    # function to find vocabulary and POS tags, as well as load the word embeddings to be used
    def preprocess_train(self, df, p=3, s=3, embedding_type='glove-wiki-gigaword-100'):
        self.embedding_type = embedding_type
        vocab = set(df['word'])
        pos_tags = set(df['pos'])
        word_vectors_all = api.load(embedding_type)

        word_vectors = {}
        for word in vocab:
            if word in word_vectors_all:
                word_vectors[word] = word_vectors_all[word]
            else:
                word_vectors[word] = torch.zeros(len(word_vectors_all['the']))

        # one hot encode the POS tags
        pos_tags_one_hot = {}
        for i, tag in enumerate(pos_tags):
            one_hot = torch.zeros(len(pos_tags) + 1)
            one_hot[i] = 1
            pos_tags_one_hot[tag] = one_hot
        pos_tags_one_hot[''] = torch.zeros(len(pos_tags) + 1)
        pos_tags_one_hot[''][-1] = 1

        # convert the df to list
        data = df.values.tolist()
        print(np.array(data).shape)
        dataset = np.array([word_vectors[data[i][1]] for i in range(len(data))], dtype=np.float32)
        # pp.pprint(dataset)

        max = 0
        for i in range(len(dataset)):
            if data[i][0] > max:
                max = data[i][0]

        split_dataset = []
        curr = 0
        for i in range(1, len(dataset)):
            if data[i][0] == 1:
                split_dataset.append(dataset[curr:i])
                curr = i
        split_dataset.append(dataset[curr:])
        # print(len(split_dataset))
        # print(max)

        final_dataset = []
        for i in range(len(split_dataset)):
            dataset = split_dataset[i]
            dataset1 = dataset.copy()
            dataset2 = dataset.copy()
            for j in range(p):
                dataset1 = dataset1[:-1]
                dataset1 = np.insert(dataset1, 0, [np.zeros(len(word_vectors_all['the']))], axis=0)
                dataset = np.hstack((dataset1[:, ], dataset), dtype=np.float32)
            for j in range(s):
                dataset2 = dataset2[1:]
                dataset2 = np.append(dataset2, [np.zeros(len(word_vectors_all['the']))], axis=0)
                dataset = np.hstack((dataset, dataset2[:, ]), dtype=np.float32)
            final_dataset.append(dataset)

        # pp.pprint(final_dataset[0][0])
        # pp.pprint(final_dataset[0][-1])

        dataset = []
        for lst in final_dataset:
            dataset.extend(lst)
        dataset = np.reshape(dataset, (len(dataset), 1, len(dataset[0])))

        final_dataset = []
        for i in range(len(dataset)):
            final_dataset.append([torch.tensor(dataset[i][0]), pos_tags_one_hot[data[i][2]]])

        print('Training Data Preprocessed')

        return final_dataset, word_vectors, pos_tags_one_hot

    # function to preprocess the dev and test data, using the word vectors and POS tags from the training data
    def preprocess_dev_test(self, df, word_vectors, pos_tags_one_hot, p=3, s=3):
        data = df.values.tolist()
        # dataset = np.array([word_vectors[data[i][1]] for i in range(len(data))], dtype=np.float32)
        dataset = []
        for i in range(len(data)):
            if data[i][1] in word_vectors:
                dataset.append(word_vectors[data[i][1]])
            else:
                dataset.append(torch.zeros(len(word_vectors['the'])))
        dataset = np.array(dataset, dtype=np.float32)
        # pp.pprint(dataset)

        max = 0
        for i in range(len(dataset)):
            if data[i][0] > max:
                max = data[i][0]

        split_dataset = []
        curr = 0
        for i in range(1, len(dataset)):
            if data[i][0] == 1:
                split_dataset.append(dataset[curr:i])
                curr = i
        split_dataset.append(dataset[curr:])
        # print(len(split_dataset))
        # print(max)

        final_dataset = []
        for i in range(len(split_dataset)):
            dataset = split_dataset[i]
            dataset1 = dataset.copy()
            dataset2 = dataset.copy()
            for j in range(p):
                dataset1 = dataset1[:-1]
                dataset1 = np.insert(dataset1, 0, [np.zeros(len(word_vectors['the']))], axis=0)
                dataset = np.hstack((dataset1[:, ], dataset), dtype=np.float32)
            for j in range(s):
                dataset2 = dataset2[1:]
                dataset2 = np.append(dataset2, [np.zeros(len(word_vectors['the']))], axis=0)
                dataset = np.hstack((dataset, dataset2[:, ]), dtype=np.float32)
            final_dataset.append(dataset)

        # pp.pprint(final_dataset[0][0])
        # pp.pprint(final_dataset[0][-1])

        dataset = []
        for lst in final_dataset:
            dataset.extend(lst)
        dataset = np.reshape(dataset, (len(dataset), 1, len(dataset[0])))

        final_dataset = []
        for i in range(len(dataset)):
            # final_dataset.append([torch.tensor(dataset[i][0]), pos_tags_one_hot[data[i][2]]])
            tensor1 = torch.tensor(dataset[i][0])
            try:
                tensor2 = pos_tags_one_hot[data[i][2]]
            except:
                tensor2 = pos_tags_one_hot['']
            final_dataset.append([tensor1, tensor2])

        print('Dev/Test Data Preprocessed')

        return final_dataset

    def train(self, epochs, train_dataloader):
        print()
        self.model.train()
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}")
            running_loss = 0.0
            for i, data in enumerate(tqdm.tqdm(train_dataloader, position=0, leave=True), 0):
                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}, loss: {running_loss/len(train_dataloader)}")

    def test(self, test_dataloader):
        self.model.eval()
        running_loss = 0.0
        total_outputs = []
        total_labels = []

        print('\nTest Set Results:\n')

        with torch.no_grad():
            for data in tqdm.tqdm(test_dataloader, position=0, leave=True):
                inputs, labels = data
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                _, actual = torch.max(labels, 1)
                outputs_copy = outputs.clone().detach().cpu().numpy()
                outputs_one_hot = np.zeros(outputs_copy.shape)
                outputs_one_hot[np.arange(outputs_copy.shape[0]), np.argmax(outputs_copy, axis=1)] = 1

                total_outputs.extend(outputs_one_hot)
                total_labels.extend(labels.clone().detach().cpu().numpy())

        total_outputs = np.array(total_outputs)
        total_labels = np.array(total_labels)

        print()
        print(f"Loss: {running_loss/len(test_dataloader)}")
        print()
        print(f"Accuracy: {accuracy_score(total_labels, total_outputs)}")
        print(f"Precision: {precision_score(total_labels, total_outputs, average='weighted', zero_division=0)}")
        print(f"Recall: {recall_score(total_labels, total_outputs, average='weighted', zero_division=0)}")
        print(f"F1 Score: {f1_score(total_labels, total_outputs, average='weighted', zero_division=0)}")

        # confusion_matrix = np.zeros((len(self.pos_tags_one_hot), len(self.pos_tags_one_hot)))
        # for i in range(len(total_labels)):
        #     actual = np.argmax(total_labels[i])
        #     predicted = np.argmax(total_outputs[i])
        #     confusion_matrix[actual][predicted] += 1
        # confusion_matrix2 = confusion_matrix / np.sum(confusion_matrix, axis=1)

        # plt.imshow(confusion_matrix)
        # plt.show()
        # plt.imshow(confusion_matrix2)
        # plt.show()

        return {
            'accuracy': accuracy_score(total_labels, total_outputs),
            'precision': precision_score(total_labels, total_outputs, average='weighted', zero_division=0),
            'recall': recall_score(total_labels, total_outputs, average='weighted', zero_division=0),
            'f1_score': f1_score(total_labels, total_outputs, average='weighted', zero_division=0)
        }

    def dev(self, dev_dataloader):
        self.model.eval()
        running_loss = 0.0
        total_outputs = []
        total_labels = []

        print('Dev Set Results:\n')

        with torch.no_grad():
            for data in tqdm.tqdm(dev_dataloader, position=0, leave=True):
                inputs, labels = data
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                _, actual = torch.max(labels, 1)
                outputs_copy = outputs.clone().detach().cpu().numpy()
                outputs_one_hot = np.zeros(outputs_copy.shape)
                outputs_one_hot[np.arange(outputs_copy.shape[0]), np.argmax(outputs_copy, axis=1)] = 1

                total_outputs.extend(outputs_one_hot)
                total_labels.extend(labels.clone().detach().cpu().numpy())

        total_outputs = np.array(total_outputs)
        total_labels = np.array(total_labels)

        print()
        print(f"Loss: {running_loss/len(dev_dataloader)}")
        print()
        print(f"Accuracy: {accuracy_score(total_labels, total_outputs)}")
        print(f"Precision: {precision_score(total_labels, total_outputs, average='weighted', zero_division=0)}")
        print(f"Recall: {recall_score(total_labels, total_outputs, average='weighted', zero_division=0)}")
        print(f"F1 Score: {f1_score(total_labels, total_outputs, average='weighted', zero_division=0)}")

        # confusion_matrix = np.zeros((len(self.pos_tags_one_hot), len(self.pos_tags_one_hot)))
        # for i in range(len(total_labels)):
        #     actual = np.argmax(total_labels[i])
        #     predicted = np.argmax(total_outputs[i])
        #     confusion_matrix[actual][predicted] += 1
        # confusion_matrix2 = confusion_matrix / np.sum(confusion_matrix, axis=1)

        # plt.imshow(confusion_matrix)
        # plt.show()
        # plt.imshow(confusion_matrix2)
        # plt.show()

        return {
            'accuracy': accuracy_score(total_labels, total_outputs),
            'precision': precision_score(total_labels, total_outputs, average='weighted', zero_division=0),
            'recall': recall_score(total_labels, total_outputs, average='weighted', zero_division=0),
            'f1_score': f1_score(total_labels, total_outputs, average='weighted', zero_division=0)
        }

    def save_model(self, path):
        data = {
            'model': self.model.state_dict(),
            'optimizer_type': self.optimizer_type,
            'optimizer': self.optimizer.state_dict(),
            'criterion_type': self.criterion_type,
            'criterion': self.criterion.state_dict(),
            'one_hot': self.pos_tags_one_hot,
            'word_vectors': self.word_vectors,
            'hidden_params': self.hidden_params,
            'embed_dim': self.embed_dim,
            'prev_n': self.prev_n,
            'succ_n': self.succ_n,
            'output_dim': self.output_dim,
            'embedding_type': self.embedding_type
        }
        torch.save(data, path)

    def load_model(self, path):
        data = torch.load(path)
        self.model = FNN(data['embed_dim'], data['prev_n'], data['succ_n'], data['hidden_params'], data['output_dim'])
        self.model.load_state_dict(data['model'])
        self.model.eval()
        self.setup_cr_op(data['criterion_type'], data['optimizer_type'])
        self.criterion_type = data['criterion']
        self.optimizer_type = data['optimizer']
        self.pos_tags_one_hot = data['one_hot']
        self.word_vectors = data['word_vectors']
        self.hidden_params = data['hidden_params']
        self.embed_dim = data['embed_dim']
        self.prev_n = data['prev_n']
        self.succ_n = data['succ_n']
        self.output_dim = data['output_dim']
        self.embedding_type = data['embedding_type']

    def predict(self, sentence):
        # check if punkt is already downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        # tokenize the sentence with nltk
        tokens = nltk.word_tokenize(sentence)

        # remove all the tokens with any non-alphabetic characters except for hyphens and apostrophes
        tokens = [token for token in tokens if token.isalpha() or (token.replace('-', '').replace('\'', '').isalpha())]

        # load the word vectors
        # word_vectors = api.load(self.embedding_type)
        word_vectors = self.word_vectors

        # get the embeddings for all the tokens
        embeddings = []
        for token in tokens:
            if token in word_vectors:
                embeddings.append(word_vectors[token])
            else:
                embeddings.append(torch.zeros(len(word_vectors['the'])))
        embeddings = np.array(embeddings, dtype=np.float32)

        # create the dataset
        dataset = []
        for i in range(len(embeddings)):
            dataset.append(embeddings[i])
        dataset = np.array(dataset, dtype=np.float32)
        dataset = np.reshape(dataset, (len(dataset), 1, len(dataset[0])))
        dataset = [torch.tensor(dataset[i][0]) for i in range(len(dataset))]
        dataset = np.array(dataset)

        final_dataset = []
        for i in range(len(dataset)):
            data_vector = []
            for p in range(self.prev_n):
                if i - p - 1 < 0:
                    data_vector.append(torch.zeros(len(list(self.word_vectors.values())[0])))
                else:
                    data_vector.append(dataset[i - p - 1])
            data_vector.append(dataset[i])
            for s in range(self.succ_n):
                if i + s + 1 >= len(dataset):
                    data_vector.append(torch.zeros(len(list(self.word_vectors.values())[0])))
                else:
                    data_vector.append(dataset[i + s + 1])
            data_vector = np.array(data_vector, dtype=np.float32)
            data_vector = np.reshape(data_vector, (1, len(data_vector)*len(data_vector[0])))
            data_vector = torch.tensor(data_vector, dtype=torch.float32, device=self.device)
            final_dataset.append(data_vector)

        # get the POS tags for the tokens
        pos_tags = []
        with torch.no_grad():
            for i in range(len(final_dataset)):
                inputs = final_dataset[i]
                outputs = self.model(inputs)
                outputs_copy = outputs.clone().detach().cpu().numpy()
                pos_tags.append(list(self.pos_tags_one_hot.keys())[np.argmax(outputs_copy)])

        for i in range(len(tokens)):
            print(f'{tokens[i]} {pos_tags[i]}')



if __name__ == '__main__':
    #import the data files
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
                data.append([token['id'], token['form'], token['upostag']])
        # return pd.DataFrame(data, columns=['', 'word', 'pos'])
        return pd.DataFrame(data, columns=['id', 'word', 'pos'])

    df_train = create_dataframe(dataset_train)
    df_dev = create_dataframe(dataset_dev)
    df_test = create_dataframe(dataset_test)

    print(len(df_train))
    print(len(df_dev))
    print(len(df_test))

    print('Dataframes Created')

    fnn_train = FNNTrainer()
    fnn_train.setup_dataloaders(df_train, df_test, df_dev, 2, 2, 'glove-wiki-gigaword-200')
    fnn_train.create_model([100, 100])
    fnn_train.setup_cr_op('bce', 'adam')
    fnn_train.train(30, fnn_train.train_dataloader)
    results_test = fnn_train.test(fnn_train.test_dataloader)
    results_dev = fnn_train.dev(fnn_train.dev_dataloader)
    fnn_train.save_model('fnn_model.pth')

    # # open a file to write the results
    # with open('results.csv', 'w') as file:
    #     # write the parameters used, along with the results of the test and dev sets, in a csv file
    #     file.write('hidden_params,accuracy_test,precision_test,recall_test,f1_score_test,accuracy_dev,precision_dev,recall_dev,f1_score_dev\n')
    #     file.write('[100, 100, 100],' + str(results_test['accuracy']) + ',' + str(results_test['precision']) + ',' + str(results_test['recall']) + ',' + str(results_test['f1_score']) + ',' + str(results_dev['accuracy']) + ',' + str(results_dev['precision']) + ',' + str(results_dev['recall']) + ',' + str(results_dev['f1_score']))
    #     file.write('\n')
    # print('Results Written to File')

    # # connecting to wandb
    # wandb.init(project='FNN_POS_Tagger_2', entity='rockingharsha71')
    # for span in range(1, 3):
    #     for embedding in ['glove-wiki-gigaword-100', 'glove-wiki-gigaword-200', 'word2vec-google-news-300']:
    #         for hidden_params in [[10, 10], [5, 5, 5], [100, 100]]:
    #             for criterion in ['bce', 'cross_entropy']:
    #                 for optimizer in ['adam']:
    #                     print('Span:', span, 'Embedding:', embedding, 'Hidden Params:', hidden_params, 'Criterion:', criterion, 'Optimizer:', optimizer)

    #                     fnn_train = FNNTrainer()
    #                     fnn_train.setup_dataloaders(df_train, df_test, df_dev, span, span, embedding)
    #                     fnn_train.create_model(hidden_params)
    #                     fnn_train.setup_cr_op(criterion, optimizer)
    #                     fnn_train.train(20, fnn_train.train_dataloader)
    #                     results_test = fnn_train.test(fnn_train.test_dataloader)
    #                     results_dev = fnn_train.dev(fnn_train.dev_dataloader)

    #                     # make each log a new run
    #                     wandb.init(project='FNN_POS_Tagger_2', entity='rockingharsha71', reinit=True)
    #                     wandb.log({
    #                         'span': span,
    #                         'embedding': embedding,
    #                         'hidden_params': hidden_params,
    #                         'criterion': criterion,
    #                         'optimizer': optimizer,
    #                         'accuracy_test': results_test['accuracy'],
    #                         'precision_test': results_test['precision'],
    #                         'recall_test': results_test['recall'],
    #                         'f1_score_test': results_test['f1_score'],
    #                         'accuracy_dev': results_dev['accuracy'],
    #                         'precision_dev': results_dev['precision'],
    #                         'recall_dev': results_dev['recall'],
    #                         'f1_score_dev': results_dev['f1_score']
    #                     })

    #                     print('Results Logged to Wandb')

    # for span in range(3, 5):
    #     for embedding in ['glove-wiki-gigaword-100', 'glove-wiki-gigaword-200']:
    #         for hidden_params in [[5, 5], [10, 10], [5, 5, 5], [100, 100]]:
    #             for criterion in ['bce', 'cross_entropy']:
    #                 for optimizer in ['adam']:
    #                     print('Span:', span, 'Embedding:', embedding, 'Hidden Params:', hidden_params, 'Criterion:', criterion, 'Optimizer:', optimizer)

    #                     fnn_train = FNNTrainer()
    #                     fnn_train.setup_dataloaders(df_train, df_test, df_dev, span, span, embedding)
    #                     fnn_train.create_model(hidden_params)
    #                     fnn_train.setup_cr_op(criterion, optimizer)
    #                     fnn_train.train(20, fnn_train.train_dataloader)
    #                     results_test = fnn_train.test(fnn_train.test_dataloader)
    #                     results_dev = fnn_train.dev(fnn_train.dev_dataloader)

    #                     # make each log a new run
    #                     wandb.init(project='FNN_POS_Tagger_2', entity='rockingharsha71', reinit=True)
    #                     wandb.log({
    #                         'span': span,
    #                         'embedding': embedding,
    #                         'hidden_params': hidden_params,
    #                         'criterion': criterion,
    #                         'optimizer': optimizer,
    #                         'accuracy_test': results_test['accuracy'],
    #                         'precision_test': results_test['precision'],
    #                         'recall_test': results_test['recall'],
    #                         'f1_score_test': results_test['f1_score'],
    #                         'accuracy_dev': results_dev['accuracy'],
    #                         'precision_dev': results_dev['precision'],
    #                         'recall_dev': results_dev['recall'],
    #                         'f1_score_dev': results_dev['f1_score']
    #                     })

    #                     print('Results Logged to Wandb')
