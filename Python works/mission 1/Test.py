import okex.spot_api as spot
import okex.futures_api as future
import time, threading
import pandas as pd
import datetime
import numpy as np

api_key = ''
seceret_key = ''
passphrase = ''

spotAPI = spot.SpotAPI(api_key, seceret_key, passphrase, True)
futureAPI = future.FutureAPI(api_key, seceret_key, passphrase, True)
## 设置定时，只运行6秒
now = datetime.datetime.now()
end_time = now +datetime.timedelta(minutes = 0.1 )
count = end_time -now
count = count.seconds
print(count)
start = 0

#设计交易规则，当期货价格大于现货价格某个点并产生交易信号时，买入现货
#定义买入的数量规则： 持续吃掉现货卖方深度，当现货上升到无套利区间时，停止买入，并计算买入总价值及购入BTC数量
def Buy_current(current_price_asks,future_last,dif,check_point):
    Total_value_buy = 0
    size_buy = 0
    #当价差值满足时，执行交易
    if dif > check_point:
        for i in range(len(current_price_asks)):
            if current_price_asks[0][i] < future_last -check_point:
                
                Total_value_buy += current_price_asks[0][i]*current_price_asks[1][i]
                size_buy += current_price_asks[1][i]
                #print(s)
                
        #print('sum of current is',format(s))
    return Total_value_buy,size_buy

while start <= count:    
    current_last = float(spotAPI.get_specific_ticker('BTC-USDT')['last'])
    future_last = float(futureAPI.get_specific_ticker('BTC-USD-191227')['last'])
    current_price_asks = pd.DataFrame.from_dict(spotAPI.get_depth('BTC-USDT',20,0.1)['asks'])
    future_bids = pd.DataFrame.from_dict(futureAPI.get_depth('BTC-USD-191227',20)['bids'])
    current_price_asks[0] = pd.to_numeric(current_price_asks[0])
    current_price_asks[1] = pd.to_numeric(current_price_asks[1])
    #这部分因为受收到的深度数据不一样，暂时没有用
    future_bids[0] = pd.to_numeric(future_bids[0])
    future_bids[1] = pd.to_numeric(future_bids[1])
    #设置的价差值
    check_point = 9
    #计算差价
    dif = future_last - current_last
    print('dif is',format(dif))
    time.sleep(1)
    
    start += 1
    Total_value_buy, size_buy= Buy_current(current_price_asks,future_last,dif,check_point)
    #由于我没办法获得跟现货order book一样精度的期货深度数据。所以通过总买入的现货两，大概估算了一下应该卖出的期货合约的价值
    #我觉得需要对冲的总价值为 现货总价值+期货溢价部分。下面的公式，我直接把溢价转换成了现货价值的一定百分比。
    #如果需要确定卖出多少期货合约，只要用期货卖出总价值除以一张合约的价格
    Total_value_sell = Total_value_buy * (current_last + check_point)/current_last
    size_sell = Total_value_sell/future_bids[0][0]
    print('Total value BTC buy is {} and total size is {}'.format( Total_value_buy,size_buy))
    print('Total value BTC_future sell is {} and total size is {}'.format( Total_value_sell,size_sell))