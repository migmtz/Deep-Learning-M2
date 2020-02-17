import torch
from torch.autograd import Function
from torch.autograd import gradcheck
from matplotlib import pyplot as plt
import numpy as np

from sklearn.datasets import load_boston
datos = load_boston()

Boston_param = torch.tensor(datos["data"],dtype = torch.float32)
Boston_y = torch.tensor(datos["target"],dtype = torch.float32)

aux = torch.randperm(506)
Boston_param = Boston_param[aux,:]
Boston_y = Boston_y[aux]


Boston_param_app = Boston_param[0:50,:].clone().detach()
Boston_param_test = Boston_param[50:506,:].clone().detach()
Boston_y_app = Boston_y[0:50].clone().detach()
Boston_y_test = Boston_y[50:506].clone().detach()


## Differentiation Automatique

EPS = 10**(-10)
EPOCH = 1000

def desc_grad_sto(param,y_a,eps,epoch):
    largo, ancho = param.size()
    w = torch.randn(1,ancho,requires_grad=True)  #Premier est c outputs, deuxieme est d parametres
    b = torch.randn(1,1,requires_grad=True)
    error = torch.zeros(epoch)
    for j in range(epoch):
        errores = torch.zeros(largo)
        for i in range(largo):
            y_hat = b + torch.mm(param[i].view(1,ancho),(w.t()))
            y = (y_a[i] - y_hat)**2
            errores[i] = y
            y.backward()
            w_aux = w.grad
            b_aux = b.grad
            w.data -= eps*w_aux
            b.data -= eps*b_aux
            w.grad.data.zero_()
            b.grad.data.zero_()
        error[j] = torch.sum(errores)/largo
    return(error)


err = desc_grad_sto(Boston_param_app,Boston_y_app,EPS,EPOCH)

plt.figure(1)
plt.plot(err.detach().numpy())
plt.xlabel("t")
plt.ylabel("Erreur")
plt.grid()
plt.show()

## Optimiseur
EPS1 = 10**(-8)
NB_EPOCH1 = 1000

def desc_grad_mb_opt(param,y_a,EPS,NB_EPOCH):
    w = torch.nn.Parameter(torch.randn(1,13))
    b = torch.nn.Parameter(torch.randn(1))
    optim = torch.optim.SGD(params=[w,b],lr=EPS) ## on optimise selon w et b, lr : pas de gradient
    optim.zero_grad()
    errores = []
    aux = 0
    # Reinitialisation du gradient
    for i in range(NB_EPOCH):
        y_hat = b + torch.mm(param[i%50].view(1,13),(w.t()))
        loss = (y_a[i%50].view((1,1)) - y_hat)**2
        aux += float(loss) #Calcul du cout
        loss.backward() # Retropropagation
        if i % 5 == 0:
            optim.step() # Mise-à-jour des paramètres w et b
            optim.zero_grad()
            errores += [aux/5]
            aux = 0
    return(errores)

c2 = desc_grad_mb_opt(Boston_param_app,Boston_y_app,EPS1,NB_EPOCH1)

plt.figure(1)
plt.plot(c2)
plt.show()

## Module
EPS1 = 10**(-3)
NB_EPOCH1 = 1000


def desc_grad_module(param,y_a,EPS,NB_EPOCH):
    largo, ancho = param.size()
    Lin1 = torch.nn.Linear(13,19)
    Lin2 = torch.nn.Linear(19,1)
    Tnh = torch.nn.Tanh()
    Mse = torch.nn.MSELoss()
    optim = torch.optim.SGD([*list(Lin1.parameters()), *list(Lin2.parameters())],lr=EPS)
    optim.zero_grad()
    errores = []
    aux = 0
    # Reinitialisation du gradient
    for i in range(NB_EPOCH):
        aux = 0
        y_hat = Lin2(Tnh(Lin1(param)))
        loss = Mse(y_hat,y_a)
        aux += float(loss)
        loss.backward()
        optim.step()
        optim.zero_grad()
        errores += [aux/largo]
    return(errores)

c3 = desc_grad_module(Boston_param_app,Boston_y_app,EPS1,NB_EPOCH1)

plt.figure(1)
plt.plot(c3)
plt.grid()
plt.show()

## Sequential

layer = torch.nn.Sequential(
    torch.nn.Linear(13, 20),
    torch.nn.Tanh(),
    torch.nn.Linear(20, 1),
)

loss_fn = torch.nn.MSELoss(reduction='mean')

l_r = 1e-3
optim = torch.optim.SGD(layer.parameters(),lr=l_r)

err = []

for i in range(1000):
    y_pred = layer(Boston_param_app)

    loss = loss_fn(y_pred, Boston_y_app)
    err += [loss]
    if i % 100 == 99:
        print(i, loss.item())

    optim.zero_grad()

    loss.backward()
    optim.step()

plt.figure(1)
plt.plot(err)
plt.grid()
plt.show()