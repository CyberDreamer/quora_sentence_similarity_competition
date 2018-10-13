import numpy as np
import pandas
import datetime
import time



data = pandas.read_csv('result_CNL+OneHot(200, 300, 400, 500).csv')
result = pandas.DataFrame(data.index, columns=['test_id'])
result['is_duplicate'] = data['is_duplicate']
result.to_csv('result.csv', index=False)




# def indexToBinaryArray(index):
#     res = np.zeros(5)
#     temp = [int(x) for x in bin(index)[2:]]
#
#     start = 5 - len(temp)
#     res[start:] = temp
#
#     return res
#
#
# index_to_code = dict((i, indexToBinaryArray(i)) for i in xrange(28))
#
# ts_1 = time.time()
# for ii in xrange(100000):
#     r = np.random.randint(0,28)
#     code = index_to_code[r]
#
# ts_2 = time.time()
# dt = ts_2 - ts_1
# print(dt)
#
# ts_1 = time.time()
# for ii in xrange(100000):
#     r = np.random.randint(0,28)
#     code = indexToBinaryArray(r)
#
# ts_2 = time.time()
# dt = ts_2 - ts_1
# print(dt)