import string
import unicodedata

LETTRES = string.ascii_letters + string.punctuation + string.digits + ' '
id2lettre = dict(zip(range(1,len(LETTRES) + 1), LETTRES))
id2lettre[0] = '' #NULLCHARACTER
lettre2id = dict(zip(id2lettre.values(), id2lettre.keys() ))

def normalize(s):
    return ''.join(c for c in unicodedata.normalize('NFD',s) if c in LETTRES)
def string2code(s):
    return torch.tensor([lettre2id[c] for c in normalize(s)])
def code2string(t):
    if type(t) != list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

##

with open('/Users/alisdair/Documents/Pyzo/M2A/Profond/TME4/AMAL-data-tp4/trump_full_speech.txt', 'r') as file:
    data = file.read()

data = string2code(data)
l = len(data)
data_embed = torch.zeros((l,96))
for i in range(l):
    data_embed[i,data[i].item()] = 1

##

d_T = 10
d_dim = 96
d_out = 10
d_batch = 1

h = torch.zeros((d_batch,d_out),dtype = torch.float32)

rnn = RNN(d_dim,d_T,d_out)

decodeur = torch.nn.Sequential(
    torch.nn.Linear(d_out,d_dim),
    torch.nn.Sigmoid()
)

l_r = 1e-3
params = list(rnn.parameters()) + list(decodeur.parameters())
optim = torch.optim.Adam(params,lr=l_r)
loss_fn = torch.nn.MSELoss(reduction='mean')

writer = SummaryWriter()

aux = 0

for N in range(50000):
    i = np.random.randint(l-d_T)
    x = data_embed[i:i+d_T,:].reshape((d_T,d_batch,d_dim))
    y = torch.tensor(data_embed[i+d_T+1])

    h_list = rnn.forward(x,h)

    y_pred = decodeur(h_list[-1])
    loss = loss_fn(y_pred,y)

    optim.zero_grad()
    loss.backward()
    optim.step()
    aux += loss
    if N%200 == 0:
        print(N,aux/200)
        aux = 0
    writer.add_scalar("Cross_Entropy_Loss_Ad_-2\Trump", loss/N, N)

print(y_pred)

##

let = 'Today I wa'
print(let,end='')
h = torch.zeros((1,d_out),dtype = torch.float32)
aux = torch.zeros((10,96))
for i in range(10):
    aux[i,lettre2id[let[i]]] = 1
for i in range(100):
    let_pred = decodeur(rnn.forward(aux.reshape((d_T,1,d_dim)),h)[-1])
    let_key = torch.argmax(let_pred)
    let = let[1:10] + id2lettre[let_key.item()]
    print(id2lettre[let_key.item()],end='')
    for j in range(9):
        aux[j,:] = aux[j+1,:]
    aux[9,:] = torch.zeros(96)
    aux[9,lettre2id[let[9]]] = 1

##

data_embed = data_embed.reshape((1339673,1,96))

d_T = 10
d_dim = 96
d_out = 15
d_batch = 10

h = torch.zeros((d_batch,d_out),dtype = torch.float32)

rnn = RNN(d_dim,d_T,d_out)

decodeur = torch.nn.Sequential(
    torch.nn.Linear(d_out,d_dim),
    torch.nn.Sigmoid()
)

l_r = 1e-2
params = list(rnn.parameters()) + list(decodeur.parameters())
optim = torch.optim.Adam(params,lr=l_r)
loss_fn = torch.nn.MSELoss(reduction='mean')

writer = SummaryWriter()

aux = 0

for N in range(25000):
    str_idx = np.random.randint(0,l-d_T-1,d_batch)
    v_idx = np.zeros(d_batch,dtype = int)
    x = batch_temp(data_embed,str_idx,v_idx,d_length+1,d_batch,d_dim)
    y = x[-1]
    x = x[0:-1]

    h_list = rnn.forward(x,h)

    y_pred = decodeur(h_list[-1])
    loss = loss_fn(y_pred,y)

    optim.zero_grad()
    loss.backward()
    optim.step()
    aux += loss
    if N%200 == 0:
        print(N,aux/200)
        aux = 0
    writer.add_scalar("Cross_Entropy_Loss_Ad_-2\Trump", loss/N, N)

print(y_pred)