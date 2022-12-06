import numpy as np
import torch
import torchvision.transforms as T


def resample_signal(input_signal, orig_f, new_f):
    resampler = T.Resample(orig_freq=orig_f, new_freq=new_f)
    input_signal = np.frombuffer(input_signal)
    input_signal = torch.from_numpy(input_signal)
    output_signal = resampler(input_signal)
    output_value = output_signal.numpy()
    output_value = output_value.tobytes()
    return output_value

gb = GearsBuilder()
gb.map(lambda x : execute('hmget', x['key'], 'signal', 's_freq', 'new_freq'))
gb.foreach(resample_signal)
gb.run('resample:*')