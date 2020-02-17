import torch
from torch.autograd import Function
from torch.autograd import gradcheck
from matplotlib import pyplot as plt
import numpy as np


class Context:
    """Very simplified context object"""
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors

class Linear1(Function):
    @staticmethod
    def forward(ctx,x,w,b):
        ctx.save_for_backward(x,w,b)
        return(torch.addmm(b,x,w.t()))

    @staticmethod
    def backward(ctx,grad_output):
        x,w,b = ctx.saved_tensors
        return(grad_output*w,grad_output*x,grad_output*torch.ones(b.size(),dtype=torch.float64)) #aqui el grad de y_hat

class mse(Function):
    @staticmethod
    def forward(ctx,y,y_hat):
        ctx.save_for_backward(y,y_hat)
        return(torch.sum((y-y_hat)**2))

    @staticmethod
    def backward(ctx,grad_output):
        y,y_hat = ctx.saved_tensors
        aux = y-y_hat
        return(2*grad_output*aux,-2*grad_output*aux)


x = torch.randn(10,5,requires_grad=True,dtype=torch.float64) #Premier est N examples, deuxieme est d
w = torch.randn(1,5,requires_grad=True,dtype=torch.float64)  #Premier est c outputs, deuxieme est d parametres
b = torch.randn(1,1,requires_grad=True,dtype=torch.float64)  #Pour faire c>1 il faut boucle, donc b (1,1)

ctx1 = Context()
ctx2 = Context()

mafonction_check = Linear1.apply
torch.autograd.gradcheck(mafonction_check,(x,w,b))


y = torch.randn(10,1,requires_grad=True,dtype=torch.float64)     #Premier est N examples
y_hat = torch.randn(10,1,requires_grad=True,dtype=torch.float64) # Deuxieme est c outputs, c = 1

mafonction1_check = mse.apply
torch.autograd.gradcheck(mafonction1_check,(y,y_hat))

##  Extraction Data

from sklearn.datasets import load_boston
datos = load_boston()

Boston_param = torch.tensor(datos["data"],dtype = torch.float64)
Boston_y = torch.tensor(datos["target"],dtype = torch.float64)

aux = torch.randperm(506)
Boston_param = Boston_param[aux,:]
Boston_y = Boston_y[aux]


Boston_param_app = Boston_param[0:50,:].clone().detach()
Boston_param_test = Boston_param[50:506,:].clone().detach()
Boston_y_app = Boston_y[0:50].clone().detach()
Boston_y_test = Boston_y[50:506].clone().detach()

##  Gradient Stochastique

N = 1000
EPS = 10**(-6)


def sto_grad2_ep(param,y,eps,w,b):
    largo, ancho = param.size()
    errores = 0
    for i in range(0,largo):
        ctx1 = Context()
        ctx2 = Context()
        errores += float(mse.forward(ctx2,y[i].view(1,1),Linear1.forward(ctx1,param[i].view(1,ancho),w,b)))
        _, aux = mse.backward(ctx2,1)
        _, d_w,d_b = Linear1.backward(ctx1,aux)
        w.data -= eps*d_w.data
        b.data -= eps*d_b.data
    errores /= largo
    return(w,b,errores)

def sto_grad_test_ep(param,y,w,b):
    largo, ancho = param.size()
    errores = 0
    for i in range(0,largo):
        ctx1 = Context()
        ctx2 = Context()
        errores += float(mse.forward(ctx2,y[i].view(1,1),Linear1.forward(ctx1,param[i].view(1,ancho),w,b)))
    errores /= largo
    return(errores)

w_0 = torch.randn(1,13,requires_grad = True, dtype=torch.float64)
b_0 = torch.randn(1,1,requires_grad = True, dtype = torch.float64)
err_app = []
err_test = []

#"with torch.no_grad() pour plus vite
for i in range(N):
    w_0,b_0,err_a = sto_grad2_ep(Boston_param_app,Boston_y_app,EPS,w_0,b_0)
    err_t = sto_grad_test_ep(Boston_param_test,Boston_y_test,w_0,b_0)
    err_app += [err_a]
    err_test += [err_t]
    if i %100 == 0:
        print(i," done")

aux = np.array([i for i in range(1,N+1)])

plt.figure(1)
plt.plot(aux,err_app,label = "Erreur App")
plt.plot(aux,err_test,label = "Erreur Test")
plt.legend()
plt.grid()
plt.show()

## Batch

N = 1000
EPS = 10**(-6)

def batch_ep(param,y,eps,w,b):
    largo, ancho = param.size()
    errores = 0
    ctx1 = Context()
    ctx2 = Context()
    errores += float(mse.forward(ctx2,y.reshape(largo,1),Linear1.forward(ctx1,param,w,b)))
    _, aux = mse.backward(ctx2,1)
    _, d_w,d_b = Linear1.backward(ctx1,aux)
    w.data -= eps*torch.sum(d_w,axis = 0)/largo
    b.data -= eps*torch.sum(d_b,axis = 0)/largo
    errores /= largo
    return(w,b,errores)

def batch_ep_test(param,y,w,b):
    largo, ancho = param.size()
    errores = 0
    ctx1 = Context()
    ctx2 = Context()
    errores += float(mse.forward(ctx2,y,Linear1.forward(ctx1,param,w,b)))/largo
    return(errores)

w_0 = torch.randn(1,13,requires_grad = True, dtype=torch.float64)
b_0 = torch.randn(1,1,requires_grad = True, dtype = torch.float64)
err_app = []
err_test = []

for i in range(N):
    w_0,b_0,err_a = batch_ep(Boston_param_app,Boston_y_app,EPS,w_0,b_0)
    err_t = sto_grad_test_ep(Boston_param_test,Boston_y_test,w_0,b_0)
    err_app += [err_a]
    err_test += [err_t]
    if i %100 == 0:
        print(i," done")

aux = np.array([i for i in range(1,N+1)])

plt.figure(2)
plt.plot(aux,err_app,label = "Erreur App")
plt.plot(aux,err_test,label = "Erreur Test")
plt.legend()
plt.grid()
plt.show()

## Mini Batch

N = 1000
EPS = 10**(-8)
mini_size = 10

def mini_batch_ep(param,y,eps,w,b,mini_size):
    largo, ancho = param.size()
    errores = 0
    ctx1 = Context()
    ctx2 = Context()
    d_w_mini = torch.zeros(w.shape,dtype = torch.float64)
    d_b_mini = torch.zeros(b.shape,dtype = torch.float64)
    for i in range(0,largo):
        errores += float(mse.forward(ctx2,y[i].view(1,1),Linear1.forward(ctx1,param[i].view(1,ancho),w,b)))
        _, aux = mse.backward(ctx2,1)
        _, d_w,d_b = Linear1.backward(ctx1,aux)
        d_w_mini += d_w
        d_b_mini += d_b
        if (i%mini_size == 0):
            w.data -= eps*d_w.data/mini_size
            b.data -= eps*d_b.data/mini_size
            d_w_mini = torch.zeros(w.shape,dtype = torch.float64)
            d_b_mini = torch.zeros(b.shape,dtype = torch.float64)
    errores /= largo
    return(w,b,errores)

def mini_batch_ep_test(param,y,w,b):
    largo, ancho = param.size()
    errores = 0
    ctx1 = Context()
    ctx2 = Context()
    for i in range(0,largo):
        errores += float(mse.forward(ctx2,y[i].view(1,1),Linear1.forward(ctx1,param[i].view(1,ancho),w,b)))
    errores /= largo
    return(errores)

w_0 = torch.randn(1,13,requires_grad = True, dtype=torch.float64)
b_0 = torch.randn(1,1,requires_grad = True, dtype = torch.float64)
err_app = []
err_test = []

for i in range(N):
    w_0,b_0,err_a = mini_batch_ep(Boston_param_app,Boston_y_app,EPS,w_0,b_0, mini_size)
    err_t = mini_batch_ep_test(Boston_param_test,Boston_y_test,w_0,b_0)
    err_app += [err_a]
    err_test += [err_t]
    if i %100 == 0:
        print(i," done")

aux = np.array([i for i in range(1,N+1)])

plt.figure(3)
plt.plot(aux,err_app,label = "Erreur App")
plt.plot(aux,err_test,label = "Erreur Test")
plt.legend()
plt.grid()
plt.show()

## Comparaison

N = 1000
EPS = 10**(-8)
mini_size = 10

w_0_or = torch.randn(1,13,requires_grad = True, dtype=torch.float64)
b_0_or = torch.randn(1,1,requires_grad = True, dtype = torch.float64)
err_app_sto = []
err_test_sto = []
err_app_bat = []
err_test_bat = []
err_app_mini = []
err_test_mini = []


w_0 = w_0_or.clone()
b_0 = b_0_or.clone()

for i in range(N):
    w_0,b_0,err_a = sto_grad2_ep(Boston_param_app,Boston_y_app,EPS,w_0,b_0)
    err_t = sto_grad_test_ep(Boston_param_test,Boston_y_test,w_0,b_0)
    err_app_sto += [err_a]
    err_test_sto += [err_t]
    if i %100 == 0:
        print(i," done sto")

w_0 = w_0_or.clone()
b_0 = b_0_or.clone()
for i in range(N):
    w_0,b_0,err_a = batch_ep(Boston_param_app,Boston_y_app,EPS,w_0,b_0)
    err_t = sto_grad_test_ep(Boston_param_test,Boston_y_test,w_0,b_0)
    err_app_bat += [err_a]
    err_test_bat += [err_t]
    if i %100 == 0:
        print(i," done batch")

w_0 = w_0_or.clone()
b_0 = b_0_or.clone()
for i in range(N):
    w_0,b_0,err_a = mini_batch_ep(Boston_param_app,Boston_y_app,EPS,w_0,b_0, mini_size)
    err_t = mini_batch_ep_test(Boston_param_test,Boston_y_test,w_0,b_0)
    err_app_mini += [err_a]
    err_test_mini += [err_t]
    if i %100 == 0:
        print(i," done mini")

aux = np.array([i for i in range(1,N+1)])

fig = plt.figure(4)
fig.suptitle("Comparaison méthodes")

ax1 = plt.subplot(1,2,1)
ax1.plot(aux,err_app_sto,label = "Stochastique")
ax1.plot(aux,err_app_bat,label = "Batch")
ax1.plot(aux,err_app_mini,label = "Mini Batch")
ax1.set_title("Apprentissage")
plt.xlabel("t")
plt.ylabel("Erreur")
ax1.legend()
ax1.grid()

ax2 = plt.subplot(1,2,2)
ax2.plot(aux,err_test_sto,label = "Stochastique")
ax2.plot(aux,err_test_bat,label = "Batch")
ax2.plot(aux,err_test_mini,label = "Mini Batch")
ax2.set_title("Test")
plt.xlabel("t")
plt.ylabel("Erreur")
ax2.legend()
ax2.grid()

plt.show()