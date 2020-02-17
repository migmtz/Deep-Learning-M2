import torch
import torchvision
import torchvision.datasets as dataset
mnist_trainset = dataset.MNIST(root='./data', train=True, download=False)
from matplotlib import pyplot as plt



## Dataset / Dataloader

class MonDataset(torch.utils.data.Dataset):
    def __init__(self,set):
        self.datos = set

    def __getitem__(self,index):
        aux = self.datos.data[index]
        aux = aux.float() / 255
        aux = aux.view(aux.shape[0]*aux.shape[1])
        return(aux,self.datos.targets[index].view(1))

    def __len__(self):
        return(len(self.datos))


mon_dataset = MonDataset(mnist_trainset)
print(mon_dataset[1][0].shape)

Batch_size = 10

data = torch.utils.data.DataLoader(MonDataset(mnist_trainset), shuffle = True, batch_size = Batch_size)
i = 0
for x,y in data:
    i += 1

print("Il y a", i, "batch de taille",Batch_size)

## Autoencodeur

class AutoEncodeur(torch.nn.Module):
    def __init__(self):
        super(AutoEncodeur,self).__init__()

        self.encodeur = torch.nn.Sequential(
            torch.nn.Linear(28*28,20),
            torch.nn.ReLU(True)
        )
        self.decodeur = torch.nn.Sequential(
            torch.nn.Linear(20,28*28),
            torch.nn.Sigmoid()
        )
    def forward(self,x):
        x = self.encodeur(x)
        x = self.decodeur(x)
        return(x)

## Mini-Batch SGD

from torch.utils.tensorboard import SummaryWriter

autoenc = AutoEncodeur()

loss_fn = torch.nn.BCELoss(reduction = "mean")
err = []
i = 0

epoch = 8

img_batch_r = torch.zeros((2*epoch, 1, 28, 28))


writer = SummaryWriter()

l_r = 1
optim = torch.optim.SGD(autoenc.parameters(),lr=l_r)

for j in range(epoch):
    for x,_ in data:
        y_pred = autoenc.forward(x)
        loss = loss_fn(y_pred,x)

        i += 1
        err += [loss/i]
        if i % 100 == 99:
            print(i, loss.item())

        optim.zero_grad()
        loss.backward()
        optim.step()

        writer.add_scalar("MSELoss\Mnist", loss, i)
    img_batch_r[j,0] = x[0].reshape(28,28)
    img_batch_r[j+epoch,0] = y_pred[0].reshape(28,28)

writer.add_images('Real', img_batch_r,0)

writer.close()


##

import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class State():
    def __init__(self,model,optim):
        self.model=model
        self.optim=optim
        self.epoch,self.iteration=0,0

if os.path.isfile("savepath"):
    with open("rb") as fp:
        state = torch.load(fp)
else:
    autoencoder = AutoEncodeur()
    autoencoder = autoencoder.to(device)
    optim = torch.optim.SGD(autoenc.parameters(),lr=l_r)
    state = State(autoencoder,optim)
for epoch in range(state.epoch,10):
    for x,y in data:
        state.optim.zero_grad()
        x = x.to(device)
        xhat = autoencoder(x)
        l = loss_fn(xhat,x)
        l.backward()
        state.optim.step()
        state.iteration += 1
    with open("wb") as fp:
        state.epoch=epoch+1
        torch.save(state,fp)
