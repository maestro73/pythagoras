import numpy as np
import torch
import pandas as pd
import time
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


start_time = time.time()
current_date = '100912-101000'

EXCHANGE = {
		'bitfinex':{
			'pair': 'BTCUSD',
			'period': ""
			},
		'binance':{
			'pair': 'BTCUSDT',
			'period': ""
			},
		'okex_spot_v3':{
			'pair': 'BTCUSDT',
			'period': ""
			},
		'huobipro':{
			'pair': 'BTCUSDT',
			'period': ""
			},
		'okex_future_v3':{
			'pair': 'BTCUSD',
			'period': "Q"
			},
		'huobidm':{
			'pair': 'BTCUSD',
			'period': "Q"
			},
		'bitmex':{
			'pair': 'BTCUSD',
			'period': "P"
			}
		}

excel_file_input = pd.ExcelFile('imbalance&return.xlsx')
input_list = []


for ex in EXCHANGE:
	input_ex = pd.read_excel(excel_file_input, ex).dropna(how='any')
	input_list.append([i[0] for i in StandardScaler().fit_transform(input_ex.loc[:, ['average imbalance']].values)])
	input_list.append([i[0] for i in StandardScaler().fit_transform(input_ex.loc[:, ['order imbalance ratio']].values)])
	input_list.append([i[0] for i in StandardScaler().fit_transform(input_ex.loc[:, ['net trade size']].values)])
	input_list.append([i[0] for i in StandardScaler().fit_transform(input_ex.loc[:, ['depth imbalance']].values)])


	# input_list.append(preprocessing.normalize(np.array(list(input_ex['average imbalance'])).reshape(1, -1))[0])
	# input_list.append(preprocessing.normalize(np.array(list(input_ex['order imbalance ratio'])).reshape(1, -1))[0])
	# input_list.append(preprocessing.normalize(np.array(list(input_ex['net trade size'])).reshape(1, -1))[0])
	# input_list.append(preprocessing.normalize(np.array(list(input_ex['depth imbalance'])).reshape(1, -1))[0])

	# input_list.append(list(input_ex['average imbalance']))
	# input_list.append(list(input_ex['order imbalance ratio']))
	# input_list.append(list(input_ex['net trade size']))
	# input_list.append(list(input_ex['depth imbalance']))
huobidm_input = pd.read_excel(excel_file_input, 'huobidm').dropna(how='any')

print('run time: ', time.time()-start_time)


inputs = np.array([list(a) for a in tuple(zip(*input_list))][:int(len(input_list[0])*0.5)], dtype='float32')
targets = np.array(list(huobidm_input['return10'])[:int(len(input_list[0])*0.5)], dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.unsqueeze(torch.from_numpy(targets), dim =1)

# print(len(inputs))
# print(len(targets))
train_ds = TensorDataset(inputs, targets)
batch_size = 5000
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
next(iter(train_dl))

model = torch.nn.Sequential(
	torch.nn.Linear(28,10),
	torch.nn.Sigmoid(),
	# torch.nn.ReLU(),
	torch.nn.Linear(10,1)
	)

# print(model.weight)
# print(model.bias)
opt = torch.optim.SGD(model.parameters(), lr=0.05)
# opt = torch.optim.Adam(model.parameters(), lr=5e-3, betas=(0.9,0.99))
loss_fn = F.mse_loss
loss = loss_fn(model(inputs), targets)


def fit(num_epochs, model, loss_fn, opt):
    for epoch in range(num_epochs):
        for xb,yb in train_dl:
        	# Generate predictions
            pred = model(xb)
            loss = loss_fn(pred, yb)
            # Perform gradient descent
            loss.backward()
            opt.step()
            opt.zero_grad()
        if epoch % 10 == 0:
        	print('Training loss: ', loss)
            # 	plt.cla()
            # 	plt.scatter(xb.data.numpy(), yb.data.numpy())
            # 	plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 2, 'color':  'red'})
            # 	plt.pause(0.5)

# plt.ion()   # 画图
# plt.show()
fit(400, model, loss_fn, opt)

inputs_out = np.array([list(a) for a in tuple(zip(*input_list))][:int(len(input_list[0])*0.5)], dtype='float32')
inputs_out = torch.from_numpy(inputs_out)
preds = list(model(inputs_out).cpu().detach().numpy())
targets = np.array(list(huobidm_input['return10'])[:int(len(input_list[0])*0.5)], dtype='float32')
bids = list(huobidm_input['bids10'][:int(len(input_list[0])*0.5)])
asks = list(huobidm_input['asks10'][:int(len(input_list[0])*0.5)])



pnl =0
pnl_random = 0
total_trade = 0
total_random_trade = 0

plt.scatter(preds, targets)
plt.show()

for i in range(len(preds)):
	if preds[i] >0.0008:
		pnl+=bids[i]
		total_trade+=1
	elif preds[i] < -0.0008:
		pnl+=asks[i]
		total_trade+=1

for i in range(len(preds)):
	if i %(int(len(preds)/total_trade)*2) == 0:
		pnl_random += bids[i]
		total_random_trade += 1
	if i %(int(len(preds)/total_trade)*2) == int(len(preds)/total_trade):
		pnl_random += asks[i]
		total_random_trade += 1



print("total pnl: ", pnl)
print("total trade amount: ", total_trade)
print("random pnl: ", pnl_random)
print("total random trade: ", total_random_trade)





