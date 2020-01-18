import okex.futures_api as future
import time
import pdb


api_key = '372af18c-b3e2-4804-98ba-0bc1dbf89982'
seceret_key = '10A6B1283C5DCB32FB03B664D9785691'
passphrase = ''

futureAPI = future.FutureAPI (api_key, seceret_key, passphrase, True)
result = futureAPI.get_products()


pdb.set_trace()

result = futureAPI.get_specific_ticker ('BTC-USD-191227')


while True:
    futureAPI = future.FutureAPI(api_key, seceret_key, passphrase, True)
    result = futureAPI.get_specific_ticker('BTC-USD-191227')
    print (result['best_bid'],result['timestamp']) 
    time.sleep( 1 )