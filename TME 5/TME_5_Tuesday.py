import torch
import gzip
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
import numpy as np

##

Batch = namedtuple("Batch", ["text", "labels"])

class TextDataset(torch.utils.data.Dataset):

    def __init__(self, text: torch.LongTensor, sizes: torch.LongTensor, labels: torch.LongTensor):
        self.text = text
        self.sizes = sizes
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.text[self.sizes[index]:self.sizes[index+1]], self.labels[index].item()

    @staticmethod
    def collate(batch):
        data = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        return Batch(torch.nn.utils.rnn.pad_sequence(data, batch_first=True), torch.LongTensor(labels))

## Dataset Text Padded

def loadata(f):
    with gzip.open(f,"rb") as fp:
        return(torch.load(fp))

train = loadata("/Users/alisdair/Documents/Pyzo/M2A/Profond/TME 5/train-1000.pth")

batch_size = 500

data_load = torch.utils.data.DataLoader(train, shuffle = True, batch_size = batch_size, collate_fn = train.collate)

i = 0
for x,y in data_load:
    i += 1

print("Nb of batch :",i)

##

Embed_layer = torch.nn.Embedding(1000,5)
a = b(a).reshape((100,5,60))
Conv_train = torch.nn.Conv1d(5,1,3,stride = 1)
c = Conv_train(a)

##

embed_size = 250

embed_layer = torch.nn.Embedding(1000,embed_size)

class Model(torch.nn.Module):
    def __init__(self,embed_size):
        super().__init__()
        self.conv_1 = torch.nn.Conv1d(embed_size,3,4,1)
        self.max_pool = torch.nn.MaxPool1d(3,2)
        self.relu = torch.nn.ReLU()
        self.linear = torch.nn.Linear(3,1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        x = self.conv_1(x.transpose(1,2))
        x = self.max_pool(x)
        x = self.relu(x)
        x = torch.max(x,2)[0]
        x = self.linear(x)
        x = self.sigmoid(x)
        return(x)

model = Model(embed_size)

l_r = 1e-4
params = list(embed_layer.parameters()) + list(model.parameters())
optim = torch.optim.Adam(params,lr=l_r)
loss_fn = torch.nn.BCELoss()

writer = SummaryWriter()

j = 1

for i in range(1,5):
    data_load = torch.utils.data.DataLoader(train, shuffle = True, batch_size = batch_size, collate_fn = train.collate)
    z = 0
    for x,y in data_load:
        x = embed_layer(x)
        y_pred = model(x)
        loss = loss_fn(y_pred,y.float())
        z += loss.item()
        optim.zero_grad()
        loss.backward()
        optim.step()
        writer.add_scalar("TME5/CNN/%s" %(i), loss, j)
        if j%100 == 0:
            print(j,loss)
        j += 1
    print("General Loss :",z)


writer.close()

##

test = loadata("/Users/alisdair/Documents/Pyzo/M2A/Profond/TME 5/test-1000.pth")

test_load = torch.utils.data.DataLoader(test, shuffle = True, batch_size = len(test), collate_fn = train.collate)

for x,y in test_load:
    x = embed_layer(x)
    y_pred = model(x)

acc = torch.sum(torch.floor(2*y_pred) == y.float().view(359,1))

print("Accuracy :",acc.item()/len(test))

##

#Les formules de recurrence sont:
# m_(i+1) = ( m_i )*( s_(i+1) )
# l_(i+1) = ( w_(i+1) - 1 )*( mi ) + l_i


# Pour retrouver les indices correspondant à une position j de sortie, on considérera que le premier indice dans une liste est bien 0, comme
# l'environnement python, dans ce cas:
# soit l et m les valeurs correspondant à la longueur et le déplacement
# intervalle = [ j*m ; j*m + l-1 ]

# Ici on a deux couches: l'une avec w = 4, s = 1 et l'autre avec w = 3, s = 2
# Alors on a l = 6 et m = 2

for i in range(len(train)):
    if i == 0: # Ceci est pour donner le cas que pour le premier example dans le train, en enlevant ceci on peut obtenir successivement les sous-séquences qui maximise les sorties
        x = train[i][0]
        x_0 = x
        x = embed_layer(x)
        x = model.conv_1(x.view(1,x.shape[1],x.shape[0]))
        x = model.max_pool(x)
        x = model.relu(x)
        x_act = torch.max(x,2)
        print(x.shape)
        print(x_act[0].shape)
        print(x_0[(x_act[1][0,0].item())*2: (x_act[1][0,0].item())*2 + 6])
        print(x_0[(x_act[1][0,1].item())*2: (x_act[1][0,1].item())*2 + 6])
        print(x_0[(x_act[1][0,2].item())*2: (x_act[1][0,2].item())*2 + 6])


