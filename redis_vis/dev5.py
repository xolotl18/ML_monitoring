import numpy as np
import redis

r = redis.Redis(host='localhost', port=6379)

signal = np.random.rand(1, 100)
signal = signal.tobytes()
mapping = {
    'signal': signal,
    's_freq': 10,
    'new_freq': 5
}

r.hset(name='resample:test', mapping=mapping)
