import ml2rt
import numpy as np
import redisai as rai
import torch

con = rai.Client(host='localhost', port=6379)
signal = np.array([1, 2, 3, 4, 5])
fft_model = ml2rt.load_model('fft.pt')

out = con.modelset('model:fft', 'torch', 'cpu', fft_model)
out1 = con.tensorset('signal:1', signal)
out2 = con.modelrun('model:fft', 'signal:1', 'fft:1')
final = con.tensorget('fft:1')
print(final)
print(torch.abs(torch.fft.rfft(torch.Tensor(signal))).numpy())
