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
        return((y-y_hat)**2)

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

## Extract Data

datos = open("/Users/alisdair/Documents/Pyzo/M2A/Profond/TME1/Boston.txt", "r")

linea = datos.readline().split()
linea = [float(x) for x in linea]
Boston_param = torch.Tensor([linea[0:13]])
Boston_y = torch.Tensor([[linea[-1]]])

linea = datos.readline().split()

while linea != []:
    linea = [float(x) for x in linea]
    aux_param = torch.tensor([linea[0:13]])
    aux_y = torch.Tensor([[linea[-1]]])
    Boston_param = torch.cat((Boston_param,aux_param))
    Boston_y = torch.cat((Boston_y,aux_y))
    linea = datos.readline().split()

datos.close()

Boston_param = torch.tensor(Boston_param,dtype=torch.float64)
Boston_y = torch.tensor(Boston_y,dtype=torch.float64)

aux = torch.randperm(506)
Boston_param = Boston_param[aux,:]
Boston_y = Boston_y[aux,:]

Boston_param_app = Boston_param[0:100,:].clone().detach()
Boston_param_test = Boston_param[100:506,:].clone().detach()
Boston_y_app = Boston_y[0:100].clone().detach()
Boston_y_test = Boston_y[100:506].clone().detach()

## Gradient Stochastique

def sto_grad(param,y,eps):
    largo, ancho = param.size()
    w = torch.randn(1,ancho,requires_grad=True,dtype=torch.float64)  #Premier est c outputs, deuxieme est d parametres
    b = torch.randn(1,1,requires_grad=True,dtype=torch.float64)
    errores = torch.ones((largo,1),dtype=torch.float64)
    for i in range(0,largo):
        ctx1 = Context()
        ctx2 = Context()
        errores[i,0] = mse.forward(ctx2,y[i].view(1,1),Linear1.forward(ctx1,param[i].view(1,ancho),w,b))
        _, aux = mse.backward(ctx2,1)
        _, d_w,d_b = Linear1.backward(ctx1,aux)
        w.data -= eps*d_w.data
        b.data -= eps*d_b.data
    return(w,b,errores)


def sto_grad2(param,y,eps,w,b):
    largo, ancho = param.size()
    errores = torch.ones((largo,1),dtype=torch.float64)
    for i in range(0,largo):
        ctx1 = Context()
        ctx2 = Context()
        errores[i,0] = mse.forward(ctx2,y[i].view(1,1),Linear1.forward(ctx1,param[i].view(1,ancho),w,b))
        _, aux = mse.backward(ctx2,1)
        _, d_w,d_b = Linear1.backward(ctx1,aux)
        w.data -= eps*d_w.data
        b.data -= eps*d_b.data
    return(w,b,errores)


w_1,b_1,c1 = sto_grad(Boston_param_app,Boston_y_app,0.0000001)

plt.figure(1)
plt.plot(c1.detach().numpy())
plt.show()

#"with torch.no_grad() pour plus vite

def sto_grad_test(param,y,w,b):
    largo, ancho = param.size()
    errores = torch.ones((largo,1),dtype=torch.float64)
    for i in range(0,largo):
        ctx1 = Context()
        ctx2 = Context()
        errores[i,0] = mse.forward(ctx2,y[i].view(1,1),Linear1.forward(ctx1,param[i].view(1,ancho),w,b))
    return(errores)

c2 = sto_grad_test(Boston_param_test,Boston_y_test,w_1,b_1)
c2 = np.cumsum(c2.detach().numpy())
c2 = c2/np.array([i for i in range(1,407)])

plt.figure(2)
plt.plot(c2)
plt.show()

##

w_0 = torch.randn(1,13,requires_grad=True,dtype=torch.float64)  #Premier est c outputs, deuxieme est d parametres
b_0 = torch.randn(1,1,requires_grad=True,dtype=torch.float64)

for i in range(1000):
    aux = torch.randperm(100)
    Boston_param_app = Boston_param_app[aux,:]
    Boston_y_app = Boston_y_app[aux,:]
    w_bis,b_bis,c_bis = sto_grad2(Boston_param_app,Boston_y_app,0.0000001,w_0,b_0)
    w_0 = w_bis
    b_0 = b_bis
    if i%999 == 0 and i != 0:
        plt.figure(1)
        plt.plot(c_bis.detach().numpy())
        plt.show()

c_test = sto_grad_test(Boston_param_test,Boston_y_test,w_0,b_0)
c_test = c_test.detach().numpy()
# c_test = np.cumsum(c_test.detach().numpy())
# c_test = c_test/np.array([i for i in range(1,407)])

plt.figure(2)
plt.plot(c_test)
plt.show()


## Epoch

N = 1000
EPS = 10**(-9)


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
    return(errores)

w_0 = torch.randn(1,13,requires_grad = True, dtype=torch.float64)
b_0 = torch.randn(1,1,requires_grad = True, dtype = torch.float64)
err_app = []
err_test = []

for i in range(N):
    w_0,b_0,err_a = sto_grad2_ep(Boston_param_app,Boston_y_app,EPS,w_0,b_0)
    err_t = sto_grad_test_ep(Boston_param_test,Boston_y_test,w_0,b_0)
    err_app += [err_a]
    err_test += [err_t]
    if i %100 == 0:
        print(i)


#"with torch.no_grad() pour plus vite


aux = np.array([i for i in range(1,1001)])

plt.figure(1)
plt.plot(aux,err_app,label = "Erreur App")
plt.plot(aux,err_test,label = "Erreur Test")
plt.legend()
plt.grid()
plt.show()

err_app = np.cumsum(err_app)
err_test = np.cumsum(err_test)
err_app = err_app/aux
err_test = err_test/aux

plt.figure(1)
plt.plot(aux,err_app,label = "Erreur App,cum")
plt.plot(aux,err_test,label = "Erreur Test,cum")
plt.legend()
plt.grid()
plt.show()

