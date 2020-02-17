import string
import unicodedata
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

LETTRES = string.ascii_letters + string.punctuation + string.digits + ' '
id2lettre = dict(zip(range(1,len(LETTRES) + 1), LETTRES))
id2lettre[0] = '' #NULLCHARACTER
lettre2id = dict(zip(id2lettre.values(), id2lettre.keys() ))

def normalize(s):
    return ''.join(c for c in unicodedata.normalize('NFD',s) if c in LETTRES)
def string2code(s):
    return torch.tensor([lettre2id[c] for c in normalize(s)])
def code2string(t):
    if type(t) != list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

##

from nltk import sent_tokenize

with open('/Users/alisdair/Documents/Pyzo/M2A/Profond/TME4/AMAL-data-tp4/trump_full_speech.txt', 'r') as file:
    file2 = normalize(file.read())
    sentences = sent_tokenize(file2)
    sentences = [[lettre2id[i] for i in j] for j in sentences]

##

class TextDataset(torch.utils.data.Dataset):

    def __init__(self, sentences: list):
        self.sentences = [torch.Tensor(i).long() for i in sentences]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index: int):
        return self.sentences[index]

    @staticmethod
    def collate(batch):
        return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)

##

txt_trump = TextDataset(sentences)

data = torch.utils.data.DataLoader(txt_trump, shuffle = True, batch_size = 100, collate_fn = txt_trump.collate)

i = 0
for x in data:
    i += 1
    print(x.shape)
print(i)

##

class LSTM(torch.nn.Module):
    def __init__(self,d_dim,d_latent):
        super().__init__()
        self.Wf = torch.nn.Sequential(
                torch.nn.Linear(d_dim+d_latent,d_latent),
                torch.nn.Sigmoid()
                )
        self.Wi = torch.nn.Sequential(
                torch.nn.Linear(d_dim+d_latent,d_latent),
                torch.nn.Sigmoid()
                )
        self.Wc = torch.nn.Sequential(
                torch.nn.Linear(d_dim+d_latent,d_latent),
                torch.nn.Tanh()
                )
        self.Wo = torch.nn.Sequential(
                torch.nn.Linear(d_dim+d_latent,d_latent),
                torch.nn.Sigmoid()
                )
        self.Tanh = torch.nn.Tanh()
        self.Ct = 0
        self.d_latent = d_latent


    def one_step(self,x,h):
        input = torch.cat((x.float(),h),1)
        f = self.Wf(input)*self.Ct
        self.Ct = f + (self.Wi(input))*(self.Wc(input))
        h_out = (self.Wo(input))*(self.Tanh(self.Ct))
        return(h_out)


    def forward(self,x):
        h_list = []
        h = torch.zeros((x.shape[1],self.d_latent),dtype = torch.float32)
        for i in range(x.shape[0]):
            h = self.one_step(x[i],h)
            h_list.append(h)
        self.Ct = 0
        return(h_list)

##

d_batch = 100
d_latent = 50
d_embed = int(np.floor(len(id2lettre)/4))

embed_layer = torch.nn.Embedding(len(id2lettre),d_embed)

lstm = LSTM(d_embed,d_latent)

decodeur = torch.nn.Sequential(
    torch.nn.Linear(d_latent,len(id2lettre)),
)

l_r = 1e-3
params = list(embed_layer.parameters()) + list(lstm.parameters()) + list(decodeur.parameters())

loss_fn = torch.nn.CrossEntropyLoss(ignore_index = 0)

writer = SummaryWriter()

j = 0

l_r_list = [1e-4]

for l_r in l_r_list:
    optim = torch.optim.Adam(params,lr=l_r)
    for epoch in range(3):
        data = torch.utils.data.DataLoader(txt_trump, shuffle = True, batch_size = d_batch, collate_fn = txt_trump.collate)
        for x in data:
            j += 1
            aux = 0
            x_2 = lstm(embed_layer(x).transpose(0,1))
            for i in range(len(x_2)-1):
                y_pred = decodeur(x_2[i])
                loss = loss_fn(y_pred,x[:,i+1])
                aux += loss
            optim.zero_grad()
            aux.backward()
            optim.step()
            writer.add_scalar("Loss_6/Trump/%s" %l_r, aux, j)
            print(j," ",end = '')

writer.close()

##

prueba = "Today"
print(prueba,end = '')
prueba = [lettre2id[i] for i in prueba]
prueba = torch.LongTensor(prueba)

for i in range(50):
    x_2 = embed_layer(prueba)
    x_2 = lstm(x_2.view(x_2.shape[0],1,x_2.shape[1]))
    y_pred = decodeur(x_2[-1])
    y_pred = torch.nn.Softmax(1)(y_pred)
    mult = torch.distributions.categorical.Categorical(probs = y_pred)
    id = mult.sample().item()
    let = id2lettre[id]
    print(let, end = '')
    prueba = torch.cat([prueba[1:-1], torch.LongTensor([id])])
