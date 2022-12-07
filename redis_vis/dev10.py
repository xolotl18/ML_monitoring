# import numpy as np
import redis
import redisai as rai

# import torch

r = redis.Redis(host='localhost', port=6379)
con = rai.Client(host='localhost', port=6379)

# print(con.tensorget('elemento_inesistente'))
print(type(r.exists('elemento_inesistente')))
