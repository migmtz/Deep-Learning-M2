import re
from pathlib import Path
from torch.utils.data import Dataset
from datamaestro import prepare_dataset
import numpy as np
import torch

EMBEDDING_SIZE = 50

ds = prepare_dataset("edu.standford.aclimdb")
word2id, embeddings = prepare_dataset('edu.standford.glove.6b.%d' % EMBEDDING_SIZE).load()

class FolderText(Dataset):
    def __init__(self, classes, tokenizer, load=False):
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = list(classes.keys())
        for label, folder in classes.items():
            for file in folder.glob("*.txt"):
                self.files.append(file)
                self.filelabels.append(label)

    def __len__(self):
        return len(self.filelabels)

    def __getitem__(self, ix):
        if type(ix) == int:
            a = self.tokenizer(self.files[ix].read_text())
        else:
            a = [self.tokenizer(i.read_text()) for i in self.files[ix]]
        return a, self.filelabels[ix]


WORDS = re.compile(r"\S+")
def tokenizer(t):
    return list([x for x in re.findall(WORDS, t.lower())])

train_data = FolderText(ds.train.classes, tokenizer, load=False)
test_data = FolderText(ds.test.classes, tokenizer, load=False)

##

class text_embedded(Dataset):
    def __init__(self,foldertext):
        self.text = foldertext

    def __len__(self):
        return(len(self.text))

    def __getitem__(self, ix):
        txt,sent = self.text[ix]
        if type(ix) == int:
            a = [torch.tensor(embeddings[[word2id.get(i) for i in txt if word2id.get(i) != None]]).float()]
            b = torch.tensor([0 if sent == 'pos' else 1]).long()
        else:
            a = [torch.tensor(embeddings[[word2id.get(i) for i in j if word2id.get(i) != None]]).float() for j in txt]
            b = torch.tensor([0 if s == 'pos' else 1 for s in sent]).long()
        return(txt,a,b)

    @staticmethod
    def collate(batch):
        uno = []
        dos
        tres
        return (txt,torch.nn.utils.rnn.pad_sequence(a, batch_first=True),b)

train_data_emb = text_embedded(train_data)

## Question 0:
from torch.utils.tensorboard import SummaryWriter

d_embed = 50

lin_layer_q = torch.nn.Sequential(
torch.nn.Linear(d_embed,1),
torch.nn.ReLU(),
torch.nn.Softmax(dim = 0)
)

lin_layer = torch.nn.Sequential(
torch.nn.Linear(50,2),
torch.nn.Sigmoid()
)
#
# dataloader = torch.utils.data.DataLoader(train_data_emb,shuffle = True, batch_size = 1)
# j = 0
# for x,y in dataloader:
#     if j == 0:
#         print(x,y)
#         j += 1

loss_fn = torch.nn.CrossEntropyLoss()

l_r = 1e-4
params = list(lin_layer_q.parameters()) + list(lin_layer.parameters())
optim = torch.optim.Adam(params,lr=l_r)

#hist_list = np.linspace(0,1000,10,dtype=int)

writer = SummaryWriter("runs/TME9AMAL")

i = 0
for epoch in range(10):
    for a,x,y in torch.utils.data.DataLoader(train_data_emb,shuffle = True, batch_size = 10, collate_fn = train_data_emb.collate):
        print(x)
        vect_q = lin_layer_q(x)
        y_aux = torch.sum((x*vect_q),dim = 0,keepdim=True)
        y_pred = lin_layer(y_aux)
        loss = loss_fn(y_pred,y)

        optim.zero_grad
        loss.backward()
        optim.step()

        if i%5000 == 0:
            print(i, loss)
            print('q',vect_q.view(1,vect_q.shape[0]))
            print(a[torch.argmax(vect_q)])

        writer.add_scalar("Question1/%s" %l_r, loss, i)
        i += 1
