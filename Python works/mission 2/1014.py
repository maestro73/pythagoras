import pandas as pd
import pdb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.svm import SVC
import time

# 合并列表
# df1 = pd.read_excel('/Users/dear/Desktop/1014/output.xlsx', index_col=0)
# df2 = pd.read_excel('/Users/dear/Desktop/1014/output3.xlsx', index_col=0)
df3 = pd.read_excel('/Users/dear/Desktop/1014/output.xlsx', index_col=0)
# df3 = df3.sample(frac=0.1, replace=False, random_state=None, axis=0)

# 提取bids20的符号
df3['sign'] = df3['bids20']/abs(df3['bids20'])df3['sign'] = df3['bids20']/abs(df3['bids20'])

# pdb.set_trace()

# 提取四个imbalance作为研究自变量
name = ['average imbalance', 'order imbalance ratio', 'net trade size', 'depth imbalance']
x_train, x_test, y_train, y_test = train_test_split(df3[name], df3['bids20'], test_size=0.3, random_state=7)

# 进行标准化，以及简单数据清洗
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
y_train.replace(to_replace=np.NaN, value=-1, inplace=True)

x_test = ss.fit_transform(x_test)
y_test.replace(to_replace=np.NaN, value=-1, inplace=True)

# 使用GradientBoosting回归
clf = GradientBoostingRegressor()

a = time.time()
model = clf.fit(x_train, y_train)
b = time.time()

print('training time: %s' % (b-a))

y_predict = clf.predict(x_test)
print(model.score(x_test, y_test))
# print(metrics.r2_score(clf.predict(x_train), y_train))

new = pd.DataFrame(y_test)
new['predict'] = y_predict

# fig = plt.figure(figsize=(17, 6))
#
# plt.plot(new.bids20)
# plt.plot(new.predict)
# plt.show()

pdb.set_trace()
