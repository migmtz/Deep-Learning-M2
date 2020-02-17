EPS1 = 0.0000001
NB_EPOCH1 = 100

def desc_grad_mb_opt(param,y_a,EPS,NB_EPOCH):
    w = torch.nn.Parameter(torch.randn(1,13,dtype=torch.float64))
    b = torch.nn.Parameter(torch.randn(1,dtype=torch.float64))
    optim = torch.optim.SGD(params=[w,b],lr=EPS) ## on optimise selon w et b, lr : pas de gradient
    optim.zero_grad()
    errores = torch.ones((20,1))
    # Reinitialisation du gradient
    for i in range(NB_EPOCH):
        y_hat = b + torch.mm(param[i].view(1,13),(w.t()))
        loss = (y_a[i].view((1,1)) - y_hat)**2 #Calcul du cout
        loss.backward() # Retropropagation
        if i % 5 == 0:
            optim.step() # Mise-à-jour des paramètres w et b
            optim.zero_grad()
            errores[int(i/5)] = loss
    return(w,b,errores)

w_2,b_2,c2 = desc_grad_mb_opt(Boston_param_app,Boston_y_app,EPS1,NB_EPOCH1)

plt.figure(1)
plt.plot(c2.detach().numpy())
plt.show()
