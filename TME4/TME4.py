import csv
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

## Extraction de data
def check_string_to_float(s):
    try:
            float(s)
            return(True)
    except:
            return(False)


data_aux = []
data = []
with open("/Users/alisdair/Documents/Pyzo/M2A/Profond/TME4/AMAL-data-tp4/tempAMAL_train.csv") as f:
    aux = csv.reader(f)
    i = 0
    for row in aux:
        if i == 0:
            villes = row[1:len(row)]
            i += 1
        else:
            data += row[1:len(row)]

for i in range(len(data)):
    if check_string_to_float(data[i]):
        data[i] = float(data[i])
    else:
        data[i] = data[i-30] # Pour les valeurs manquantes, on les fixe à 0

data = np.reshape(np.array(data),(11115,30))
data = torch.from_numpy(data)
data = data.float()

min = torch.min(data)
max = torch.max(data)

data = (data-min)/(max-min)

dix_data = data[:,0:10].clone().reshape((11115,10,1))

data = data.reshape((11115,30,1))


## RNN

class RNN(torch.nn.Module):
    def __init__(self,d_dim,d_length,d_latent):
        super(RNN,self).__init__()

        self.cache = torch.nn.Sequential(
            torch.nn.Linear(d_dim + d_latent,d_latent),
            torch.nn.Tanh()
        )

    def one_step(self,x,h):
        return(self.cache(torch.cat((x,h),1)))

    def forward(self,x,h):
        h_list = []
        for i in range(x.shape[0]):
            h = self.one_step(x[i],h)
            h_list.append(h)
        return(h_list)

##

def batch_temp(data,heure,v,d_length,d_batch,d_dim):
    a = torch.empty((d_length,d_batch,d_dim))
    for i in range(d_batch):
        a[:,i,:] = data[heure[i]:heure[i]+d_length,v[i].item()]
    return(a)

## Training

nb_heures = dix_data.shape[0]
nb_epoch = 100000

d_length = 10
d_batch = 100
d_dim = 1
d_latent = 20

rnn = RNN(d_dim,d_length,d_latent)

h = torch.zeros((d_batch,d_latent),dtype = torch.float32)

decodeur = torch.nn.Sequential(
    torch.nn.Linear(d_latent,10),
    torch.nn.Sigmoid()
)

l_r = 3e-3
params = list(rnn.parameters()) + list(decodeur.parameters())
optim = torch.optim.Adam(params,lr=l_r)
loss_fn = torch.nn.CrossEntropyLoss()

writer = SummaryWriter()

aux = 0

for N in range(nb_epoch):
    v_idx = np.random.randint(0,10,d_batch)
    heure_idx = np.random.randint(0,nb_heures-d_length,d_batch) #Random choice of temperatures

    x = batch_temp(dix_data,heure_idx,v_idx,d_length,d_batch,d_dim)

    h_list = rnn.forward(x,h)
    y = torch.tensor(v_idx,dtype = torch.long)

    y_pred = decodeur(h_list[-1])
    loss = loss_fn(y_pred,y)

    optim.zero_grad()
    loss.backward()
    optim.step()
    aux += loss
    if N%2000 == 0:
        print(N,aux/2000)
        aux = 0
    writer.add_scalar("Cross_Entropy_Loss_Ad_-3\Temperatures", loss/N, N)


writer.close()

# Softmax + Log(Crossentropie) (Logloss??)

#Pour Forecast, soit des RNN individuels, soit un RNN multivarié

## Test Accuracy

acc = 0

h = torch.zeros((1,d_latent),dtype = torch.float32)

freq = [0 for i in range(10)]
perc = [0 for i in range(10)]

for N in range(5000):
    i = np.random.randint(0,10)
    idx = np.random.randint(0,nb_heures-d_length)

    x = dix_data[idx:idx+d_length,i].reshape((d_length,1,d_dim))

    h_list = rnn.forward(x,h)
    y = torch.tensor([i],dtype = torch.long)

    y_pred = decodeur(h_list[-1])
    prediction = torch.argmax(y_pred).item()

    freq[i] += 1
    if prediction == i:
        acc += 1
        perc[i] += 1

print("Accuracy per class: ",[format(perc[i]/freq[i], '.2f') for i in range(10)])
print("General Accuracy:",acc/5000)

## Prediction univariée

i = np.random.randint(0,30)
print("Ville choisie %s : %s" % (i, villes[i]))

data_ville = data[:,i,:].reshape(11115,1,1)

nb_epoch = 50000

d_length = 10
d_batch = 100
d_dim = 1
d_latent = 20

rnn_pred = RNN(d_dim,d_length,d_latent)
h = torch.zeros((d_batch,d_latent),dtype = torch.float32)

decodeur = torch.nn.Sequential(
    torch.nn.Linear(d_latent,1),
    torch.nn.Sigmoid()
)

l_r = 3e-5
params = list(rnn_pred.parameters()) + list(decodeur.parameters())
optim = torch.optim.Adam(params,lr=l_r)
loss_fn = torch.nn.MSELoss(reduction='mean')

for N in range(nb_epoch):
    v_idx = np.zeros(d_batch,dtype = int)
    heure_idx = np.random.randint(0,nb_heures-d_length,d_batch)
    x = batch_temp(data_ville,heure_idx,v_idx,d_length+1,d_batch,d_dim)
    y = x[-1]
    x = x[0:-1]

    h_list = rnn_pred.forward(x,h)
    y_pred = decodeur(h_list[-1])

    loss = loss_fn(y_pred,y)

    optim.zero_grad()
    loss.backward()
    optim.step()
    aux += loss
    if N%50000 == 0:
        print(N,aux/5000)
        aux = 0

## Prediction

err = []
h = torch.zeros((1,d_latent),dtype = torch.float32)

for N in range(5000):
    i = np.random.randint(0,nb_heures-d_length)

    x = data_ville[i:i+d_length+1].reshape((d_length+1,1,d_dim))
    y = x[-1]
    x = x[0:-1]

    h_list = rnn.forward(x,h)

    y_pred = decodeur(h_list[-1])

    err += [loss_fn(y_pred,y).item()]
    if N%1000 == 0:
        print(i,"Observé: ",y.item()*(max-min) + min,"Prediction: ",y_pred.item()*(max-min) + min)

print("Erreur: ",np.mean(err))