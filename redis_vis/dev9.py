import numpy as np
import redis
import redisai as rai

# import torch

r = redis.Redis(host='localhost', port=6379)
con = rai.Client(host='localhost', port=6379)

filename = 0

sampling_freq = int(r.hget(name='signal:'+str(filename),
                           key='sampling_f'))
'''
signal_x = np.frombuffer(r.hget(name="signal:"+str(filename), key='x'),
                         dtype=np.float32)
signal_y = np.frombuffer(r.hget(name="signal:"+str(filename), key='y'),
                         dtype=np.float32)
signal_z = np.frombuffer(r.hget(name="signal:"+str(filename), key='z'),
                         dtype=np.float32)

stacked = np.stack([signal_x, signal_y, signal_z], axis=0)
print(stacked.shape)
'''
signals = r.hmget(name="signal:"+str(filename), keys=['x', 'y', 'z'])
np_signals = []
for signal in signals:
    np_signals.append(np.frombuffer(signal, dtype=np.float32))

stacked = np.stack(np_signals, axis=0)
# print(stacked.shape)
con.tensorset('tensor:stacked', stacked)
con.modelrun('model:fft', 'tensor:stacked', 'fft:stacked')
output_tensor = con.tensorget('fft:stacked')

print(output_tensor.shape)
