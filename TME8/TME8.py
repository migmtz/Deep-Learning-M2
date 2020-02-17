from datamaestro import prepare_dataset
import torch
import torchvision
import torchvision.datasets as dataset
mnist_trainset = dataset.MNIST(root='./data', train=True, download=False)
from matplotlib import pyplot as plt
from collections import OrderedDict
import numpy as np
from torch.utils.tensorboard import SummaryWriter


ds = prepare_dataset("com.lecun.mnist")
train_images, train_labels = ds.files["train/images"].data(), ds.files["train/labels"].data()
test_images, test_labels = ds.files["test/images"].data(), ds.files["test/labels"].data()

class MonDataset(torch.utils.data.Dataset):
    def __init__(self,set_data,set_labels):
        self.datos = torch.tensor(set_data)
        self.label = torch.tensor(set_labels)

    def __getitem__(self,index):
        aux = self.datos[index]
        aux = aux.float() / 255
        aux = aux.view(aux.shape[0]*aux.shape[1])
        return(aux,self.label[index].view(1))

    def __len__(self):
        return(len(self.datos))

mon_dataset = MonDataset(train_images[0:3000],train_labels[0:3000])
print(mon_dataset[1][0].shape[0])

##

# from torch.utils.tensorboard import SummaryWriter
# import numpy as np
# writer = SummaryWriter("runs/TME8AMAL")
# for i in range(10):
#     x = fc_layer1.fc1.weight
#     writer.add_histogram('distribution centers', x, i)
# writer.close()

##

fc_layer = torch.nn.Sequential(OrderedDict([
("fc1",torch.nn.Linear(mon_dataset[1][0].shape[0], 100)),
("relu1",torch.nn.ReLU()),
("fc2",torch.nn.Linear(100,100)),
("relu2",torch.nn.ReLU()),
("fc3",torch.nn.Linear(100,100)),
("relu3",torch.nn.ReLU()),
("clas",torch.nn.Linear(100,10)),
]))

# class FC_Layer(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = torch.nn.Linear(mon_dataset[1][0].shape[0], 100)
#         self.fc2 = torch.nn.Linear(100,100)
#         self.fc3 = torch.nn.Linear(100,100)
#         self.clas = torch.nn.Linear(100,10)
#         self.relu = torch.nn.ReLU()
#         self.dropout = torch.nn.Dropout(0.2)
#
#     def forward(self,x):
#         y = self.fc1(x)
#         y = self.relu(y)
#         y = self.dropout(y)
#         y = self.fc2(y)
#         y = self.relu(y)
#         y = self.dropout(y)
#         y = self.fc3(y)
#         y = self.relu(y)
#         y = self.dropout(y)
#         y = self.clas(y)
#         return(y)
#
# fc_layer = FC_Layer()

d_batch = 300
nb_epoch = 1000

loss_fn = torch.nn.CrossEntropyLoss()

l_r = 1e-3
optim = torch.optim.Adam(fc_layer.parameters(),lr=l_r)

hist_list = np.linspace(0,1000,10,dtype=int)

writer = SummaryWriter("runs/TME8AMAL")
j = 0


for epoch in range(nb_epoch):
    data = torch.utils.data.DataLoader(mon_dataset, shuffle = True, batch_size = d_batch)
    loss_aux = []
    for x,y in data:
        y_pred = fc_layer(x)
        loss = loss_fn(y_pred,y.long().view(d_batch))
        loss_aux += [loss.item()]

        optim.zero_grad()
        loss.backward()
        optim.step()


    writer.add_scalar("loss l_r %s" %(l_r), np.mean(loss_aux), epoch)

    if epoch in hist_list:
        writer.add_histogram('fc1', fc_layer.fc1.weight, j)
        writer.add_histogram('fc2', fc_layer.fc2.weight, j)
        writer.add_histogram('fc3', fc_layer.fc3.weight, j)
        writer.add_histogram('clas', fc_layer.clas.weight, j)
        j += 1

    if epoch%50 == 0:
        print("Epoch number:",epoch,"with error",np.mean(loss_aux))


writer.close()




































