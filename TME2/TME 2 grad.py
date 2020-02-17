import torch

def desc_grad_sto(param,y_a,eps):
    largo, ancho = param.size()
    w = torch.randn(1,ancho,requires_grad=True,dtype=torch.float64)  #Premier est c outputs, deuxieme est d parametres
    b = torch.randn(1,1,requires_grad=True,dtype=torch.float64)
    errores = torch.ones((largo,1))
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
    return(w,b,errores)

w_r,b_r,cr = desc_grad_sto(Boston_param_app,Boston_y_app,0.0000001)

plt.figure(1)
plt.plot(cr.detach().numpy())
plt.show()
