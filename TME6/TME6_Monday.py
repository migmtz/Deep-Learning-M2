import torch

class NN(torch.nn.Module):
    def __init__(self, inSize, outSize, layers=[], Softmax = False):
        super(NN, self).__init__()
        self.layers = torch.nn.ModuleList([])
        for x in layers:
            self.layers.append(torch.nn.Linear(inSize, x))
            inSize = x
            self.layers.append(torch.nn.Linear(inSize, outSize))
        if Softmax:
            self.layers.append(torch.nn.Softmax())
    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = torch.nn.functional.leaky_relu(x)
            x = self.layers[i](x)
        return x

class PPO_KL(object):
    def __init__(self, action_space,inSize,outSize,layer,beta = 0.2, gamma = 0.999, delta = 0.5, K = 10):
        self.action_space = action_space
        self.V = NN(inSize,outSize,[layer])
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.K = K
        self.compteur =