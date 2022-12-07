from io import BytesIO

import ml2rt
import numpy as np
import redis
import redisai as rai
import torch
import torchaudio.transforms as tat

r = redis.Redis(host='localhost', port=6379)
con = rai.Client(host='localhost', port=6379)

transforms_list = []
transforms_list.append(tat.AmplitudeToDB())

transforms = torch.nn.Sequential(*transforms_list)

# load the transform model to redisai and run it to obtain the output
# spectrogram

transform_model = torch.jit.script(transforms)
# provvisorio
fp = BytesIO()

torch.jit.save(transform_model, fp)

fp.seek(0)
model = fp.read()
fp.seek(0)
model2 = ml2rt.load_model('model.pt')
signal_x = np.random.rand(1, 100)

con.modelstore('model:test', 'torch', 'cpu', data=model)
con.tensorset('tensor:test', signal_x)
con.modelrun('model:test', 'tensor:test', 'spectrogram:test')
print(con.tensorget('spectrogram:test'))

# print(model)
# print()
# print(model2)
# transform_model.save('model1.pt')
