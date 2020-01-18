import okex.spot_api as spot
import okex.futures_api as future
import pandas as pd

api_key = ''
seceret_key = ''
passphrase = ''
check_point_1 = 20
check_point_2 = 5

# 获取现货数据
spotAPI = spot.SpotAPI(api_key, seceret_key, passphrase, True)
current_last = float(spotAPI.get_specific_ticker('BTC-USDT')['last'])
current_price_asks = pd.DataFrame.from_dict(spotAPI.get_depth('BTC-USDT', 20)['asks'])
current_price_asks[0] = pd.to_numeric(current_price_asks[0])
current_price_asks[1] = pd.to_numeric(current_price_asks[1])
current_price_bids = pd.DataFrame.from_dict(spotAPI.get_depth('BTC-USDT', 20)['bids'])
current_price_bids[0] = pd.to_numeric(current_price_bids[0])
current_price_bids[1] = pd.to_numeric(current_price_bids[1])

# 获取期货数据
futureAPI = future.FutureAPI(api_key, seceret_key, passphrase, True)
future_bids = pd.DataFrame.from_dict(futureAPI.get_depth('BTC-USD-191227', 20)['bids'])
future_bids[0] = pd.to_numeric(future_bids[0])
future_bids[1] = pd.to_numeric(future_bids[1])
future_asks = pd.DataFrame.from_dict(futureAPI.get_depth('BTC-USD-191227', 20)['asks'])
future_asks[0] = pd.to_numeric(future_asks[0])
future_asks[1] = pd.to_numeric(future_asks[1])



# 做多现货，做空期货的情况
if future_bids[0][0] > current_price_asks[0][0] + check_point_1:
    print(futureAPI.get_specific_ticker('BTC-USD-191227')['timestamp'])
    Total_value_buy = 0
    size_buy = 0
    for i in range(len(current_price_asks)):
        if future_bids[0][0] > current_price_asks[0][i] + check_point_1:
            print('Buy BTC_current with price {} and size {}'.format(current_price_asks[0][i], current_price_asks[1][i]))
            Total_value_buy += current_price_asks[0][i]*current_price_asks[1][i]
            size_buy += current_price_asks[1][i]
    Total_future_sell = Total_value_buy * (current_last + check_point_1)/current_last
    size_future_sell = Total_future_sell/future_bids[0][0]
    print('Total value BTC_current buy value is {} and total size is {}'.format(Total_value_buy, size_buy))
    print('Total value BTC_future sell value is {} and total size is {}'.format(Total_future_sell, size_future_sell))

# 做空现货，做多期货的情况
if future_asks[0][0] < current_price_bids[0][0] + check_point_2:
    print(futureAPI.get_specific_ticker('BTC-USD-191227')['timestamp'])
    Total_value_sell = 0
    size_sell = 0
    for i in range(len(current_price_bids)):
        if future_asks[0][0] < current_price_bids[0][i] + check_point_2:
            print('Sell BTC_current with price {} and size {}'.format(current_price_bids[0][i], current_price_bids[1][i]))
            Total_value_sell += current_price_bids[0][i]*current_price_bids[1][i]
            size_sell += current_price_bids[1][i]
    Total_future_buy = Total_value_sell * (current_last + check_point_2)/current_last
    size_future_buy = Total_future_buy/future_asks[0][0]
    print('Total value BTC_current buy value is {} and total size is {}'.format(Total_value_sell, size_sell))
    print('Total value BTC_future sell value is {} and total size is {}'.format(Total_future_buy, size_future_buy))
