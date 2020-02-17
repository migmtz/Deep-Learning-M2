import torch
from torch.autograd import Function
from torch.autograd import gradcheck

a = torch.rand((1,10),requires_grad=True)
b = torch.rand((1,10),requires_grad=True)
c = a.mm(b.t())
d = 2 * c
c.retain_grad() # on veut conserver le gradient par rapport à c
d.backward() ## calcul du gradient et retropropagation jusqu’aux feuilles du graphe de calcul
print(d.grad) #Rien : le gradient par rapport à d n’est pas conservé
print(c.grad) # Celui-ci est conservé
print(a.grad) ## gradient de c par rapport à a qui est une feuille
print(b.grad) ## gradient de c par rapport à b qui est une feuille

# with torch.no_grad():
#     c = a.mm(b.t())
# c.backward() ## Erreur