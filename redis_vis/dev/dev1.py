import numpy as np
import pandas as pd
import redis

r = redis.Redis(host='localhost', port=6379)

filename = 0

x_axis = np.frombuffer(r.hget(name="signal:"+str(filename), key='x'), dtype=np.float32)
y_axis = np.frombuffer(r.hget(name="signal:"+str(filename), key='y'), dtype=np.float32)
z_axis = np.frombuffer(r.hget(name="signal:"+str(filename), key='z'), dtype=np.float32)

signal = {
    'x': x_axis,
    'y': y_axis,
    'z': z_axis
}
data_chunk = pd.DataFrame(signal)
print(np.array(data_chunk['x']))


