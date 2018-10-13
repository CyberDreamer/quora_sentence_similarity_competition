import numpy as np
import time
import datetime
import pickle_file_manager




# N = 10000
# n = 100
# a = np.zeros((N, n))
# b = np.ones((N, n))

a = np.random.rand(20000, 20000)

ts_1 = time.time()
# for i in xrange(N):
#     b[i,:] = a[i, :]

b = a[0,:].sum()

ts_2 = time.time()
dt = ts_2 - ts_1
print(dt)

# a = np.zeros((n, N))
# b = np.ones((n, N))
ts_1 = time.time()
# for i in xrange(N):
#     b[i,:] = np.compress([0, i], a, axis=0)

c = a[:,0].sum()

ts_2 = time.time()
dt = ts_2 - ts_1
print(dt)

exit()




import ctypes

dllPath = 'C# DLL/PythonService/QuoraMath/bin/Debug/QuoraMath.dll'
# dllPath = 'QuoraMath.dll'
a = ctypes.cdll.LoadLibrary(dllPath)
a.add(3, 5)