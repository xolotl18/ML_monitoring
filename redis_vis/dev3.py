# FONDAMENTALE CREARE IL MODELLO CON LA STESSA VERSIONE DI TORCHAUDIO
# PRESENTE IN REDISAI. PER REDISAI 1.2.7 TORCH==1.11.0 QUINDI TORCHAUDIO 0.11.0

import argparse
import time

import ml2rt
# import numpy as np
import redisai as rai
import torch
import torchaudio.transforms as T

con = rai.Client(host='localhost', port=6379)

parser = argparse.ArgumentParser()
parser.add_argument('new_freq')
args = parser.parse_args()
new_freq = int(args.new_freq)

input_tensor = torch.randn(1, 160)

f = T.Resample(orig_freq=10, new_freq=new_freq)
scripted_model = torch.jit.script(f)
start = time.time()
print(scripted_model(input_tensor))
scripted_model.save('test.pt')

loaded_model = ml2rt.load_model('test.pt')

#con.modelstore('model:1', 'torch', 'cpu', data=scripted_model)
