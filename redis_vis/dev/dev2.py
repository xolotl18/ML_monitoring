# misura tempo per fare tensorset, modelrun e tensorget da redis
# misura differenza di tempo di esecuzione tra np.tobytes/frombuffer e pickle

import time
from statistics import mean

import numpy as np
import pandas as pd
import redis
import redisai as rai

con = rai.Client(host='localhost', port=6379)
r = redis.Redis(host='localhost', port=6379)

fetch_times = []
tensorset_times = []
modelrun_times = []
tensorget_times = []
tot_times = []

FILES = []
for file in r.keys(pattern='signal:*'):
    FILES.append(int(file.decode('utf-8').split(':')[-1]))
FILES.sort()

for filename in FILES:
    start = time.time()
    # getting signals from redis db
    signal_x = np.frombuffer(r.hget(name="signal:"+str(filename), key='x'),
                             dtype=np.float32)
    signal_y = np.frombuffer(r.hget(name="signal:"+str(filename), key='y'),
                             dtype=np.float32)
    signal_z = np.frombuffer(r.hget(name="signal:"+str(filename), key='z'),
                             dtype=np.float32)
    t1 = time.time()
    fetch_times.append(t1-start)
    # saving signals as tensors in redis db
    outx = con.tensorset('tensor:'+str(filename)+':x', signal_x)
    outy = con.tensorset('tensor:'+str(filename)+':y', signal_y)
    outz = con.tensorset('tensor:'+str(filename)+':z', signal_z)
    t2 = time.time()
    tensorset_times.append(t2-t1)
    # running the model with redisai
    outx1 = con.modelrun('model:fft', 'tensor:'+str(filename)+':x',
                         'fft:'+str(filename)+':x')
    outy1 = con.modelrun('model:fft', 'tensor:'+str(filename)+':y',
                         'fft:'+str(filename)+':y')
    outz1 = con.modelrun('model:fft', 'tensor:'+str(filename)+':z',
                         'fft:'+str(filename)+':z')
    t3 = time.time()
    modelrun_times.append(t3-t2)
    # get the output of the model from redis db
    fft_x = con.tensorget('fft:'+str(filename)+':x')
    fft_y = con.tensorget('fft:'+str(filename)+':y')
    fft_z = con.tensorget('fft:'+str(filename)+':z')
    fft_all = {
        'x': fft_x,
        'y': fft_y,
        'z': fft_z
    }
    t4 = time.time()
    tensorget_times.append(t4-t3)
    # end of data fetching and processing
    tot_times.append(t4-start)

times = {
    'total': mean(tot_times),
    'fetch': mean(fetch_times),
    'tensorset': mean(tensorset_times),
    'modelrun': mean(modelrun_times),
    'tensorget': mean(tensorget_times)
}

times_df = pd.DataFrame(times.items(), columns=['operation', 'time'])
times_df.to_csv('latencies.csv')
