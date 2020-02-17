r_t = torch.nn.CrossEntropyLoss(reduction = 'none')(torch.nn.utils.rnn.pad_packed_sequence(torch.nn.LSTM(10584,21)(a)[0],batch_first=True)[0].permute(0,2,1),y_0)
print(r_t)
r = y_0 != 20
print(r)
print(torch.masked_select(r_t,y_0 != 20))