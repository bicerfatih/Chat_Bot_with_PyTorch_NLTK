import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet



with open('intents.json', 'r') as f:
    intents = json.load(f)

print(intents)

words_all= []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        words_all.extend(w) #we use extend instead of append because we dont want to put array in an array.
        xy.append((w, tag))


ignore_words = ['?', '!', '.', ',']
words_all = [stem(w) for w in words_all if w not in ignore_words] #list comprehension
words_all = sorted(set(words_all))
tags = sorted(set(tags))
#print(words_all)
#print(tags)


#Creating the training data

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, words_all)
    X_train.append(bag)

    label = tags.index(tag) #with the indexing we will have numbers for labels
    y_train.append(label) #CrossEntropyLoss


X_train = np.array(X_train)
y_train = np.array(y_train)

print(X_train)
print(y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

        # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
#print(input_size, len(words_all))
#print(output_size, tags)
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size = batch_size, shuffle = True, num_workers = 0)


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model = NeuralNet(input_size, hidden_size, output_size).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "words_all": words_all,
    "tags":tags

}

FILE = "data2.pth"
torch.save(data, FILE)

print (f'training complete. File saved to {FILE}')