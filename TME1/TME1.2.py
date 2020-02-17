import torch
from torch.autograd import Function
from torch.autograd import gradcheck
from datamaestro import prepare_dataset

class Context:
    """Very simplified context object"""
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors


class MaFonction(Function):
    @staticmethod
    def forward(ctx,input):
        ctx.save_for_backward(input)
        return None

    @staticmethod
    def backward(ctx,grad_output):
        input = ctx.saved_tensors
        return None


## Exemple d'implementation de fonction a 2 entrÃ©es
class MaFonction:
    @staticmethod
    def forward(ctx,x,w):
        ctx.save_for_backward(x,w)
        return x+w

    @staticmethod
    def backward(ctx, grad_output):
        x,w = ctx.saved_tensors
        return x+grad_output, w+grad_output

## Pour utiliser la fonction
mafonction = MaFonction()
ctx = Context()
output = mafonction.forward(ctx,x,w)
mafonction_grad = mafonction.backward(ctx,1)

## Pour tester le gradient
mafonction_check = MaFonction.apply
x = torch.randn(10,5,requires_grad=True,dtype=torch.float64)
w = torch.randn(1,5,requires_grad=True,dtype=torch.float64)
torch.autograd.gradcheck(mafonction_check,(x,w))

## Pour telecharger le dataset Boston
ds=prepare_dataset("edu.uci.boston")
fields, data =ds.files.data()