import numpy as np
import pandas as pd
import panel as pn
import param
import redis
import redisai as rai
import torch
import torchaudio.transforms as T

pd.options.plotting.backend = "hvplot"

r = redis.Redis(host='localhost', port=6379)
con = rai.Client(host='localhost', port=6379)

FILES = []
for file in r.keys(pattern='signal:*'):
    FILES.append(int(file.decode('utf-8').split(':')[-1]))
FILES.sort()


class Data(param.Parameterized):
    files = param.Selector(FILES)

    @staticmethod
    def get_df(filename):
        # get the data from redis and put it in a dataframe with columns
        # x, y, z and time
        x_axis = np.frombuffer(r.hget(name="signal:"+str(filename), key='x'),
                               dtype=np.float32)
        y_axis = np.frombuffer(r.hget(name="signal:"+str(filename), key='y'),
                               dtype=np.float32)
        z_axis = np.frombuffer(r.hget(name="signal:"+str(filename), key='z'),
                               dtype=np.float32)

        signal = {
            'x': x_axis,
            'y': y_axis,
            'z': z_axis
        }
        data_chunk = pd.DataFrame(signal)

        return data_chunk


class Waveform(Data):
    duration_in_seconds = param.Selector([1, 5, 10, 'full'])

    def view_waveform(self):
        filename = str(self.files)
        data_df = self.get_df(self.files)
        sampling_freq = int(r.hget(name='signal:'+filename, key='sampling_f'))
        if self.duration_in_seconds == 'full':
            data_chunk = data_df.copy()
        else:
            duration_in_samples = int(int(self.duration_in_seconds) *
                                      sampling_freq)
            data_chunk = data_df[:duration_in_samples].copy()

        data_chunk['time'] = data_chunk.index / sampling_freq
        data_plot = data_chunk.plot(
            subplots=True,
            width=300,
            x='time',
            xlabel='Time (s)',
            ylabel='Amplitude',
            shared_axes=True

        )

        status = r.hget(name='signal:'+filename, key='status')
        status_panel = pn.pane.Str(f"Status: {status.decode('utf-8')}")

        row = pn.Row(data_plot, status_panel)
        return row


class ResampledWaveform(Data):
    duration_in_seconds = param.Selector([1, 5, 10, 'full'])
    resampling_freq = param.Integer(1000, bounds=(100, 1500))

    @staticmethod
    def get_resampler(orig_freq, new_freq):
        resampler = T.Resample(orig_freq=orig_freq, new_freq=new_freq)
        return torch.jit.script(resampler)

    def view_waveform(self):
        filename = str(self.files)
        sampling_freq = int(r.hget(name='signal:'+filename, key='sampling_f'))

        signal_x = np.frombuffer(r.hget(name="signal:"+str(filename), key='x'),
                                 dtype=np.float32)
        signal_y = np.frombuffer(r.hget(name="signal:"+str(filename), key='y'),
                                 dtype=np.float32)
        signal_z = np.frombuffer(r.hget(name="signal:"+str(filename), key='z'),
                                 dtype=np.float32)

        outx = con.tensorset('tensor:'+str(filename)+':x', signal_x)
        outy = con.tensorset('tensor:'+str(filename)+':y', signal_y)
        outz = con.tensorset('tensor:'+str(filename)+':z', signal_z)

        if self.resampling_freq == sampling_freq:
            resampling_f = sampling_freq+1
        else:
            resampling_f = self.resampling_freq
        resampling_model = self.get_resampler(sampling_freq, resampling_f)

        con.modelset(key='model:resampling:'+str(self.resampling_freq),
                     backend='torch', device='cpu', data=resampling_model)

        outx1 = con.modelrun('model:resampling:'+str(self.resampling_freq),
                             'tensor:'+str(filename)+':x',
                             'resampled:'+str(filename)+':x')
        outy1 = con.modelrun('model:resampling:'+str(self.resampling_freq),
                             'tensor:'+str(filename)+':y',
                             'resampled:'+str(filename)+':y')
        outz1 = con.modelrun('model:resampling:'+str(self.resampling_freq),
                             'tensor:'+str(filename)+':z',
                             'resampled:'+str(filename)+':z')

        resampled_x = con.tensorget('resampled:'+str(filename)+':x')
        resampled_y = con.tensorget('resampled:'+str(filename)+':y')
        resampled_z = con.tensorget('resampled:'+str(filename)+':z')
        resampled_all = {
            'x': resampled_x,
            'y': resampled_y,
            'z': resampled_z
        }
        data_df = pd.DataFrame(resampled_all)

        if self.duration_in_seconds == 'full':
            data_chunk = data_df.copy()
        else:
            duration_in_samples = int(int(self.duration_in_seconds) *
                                      sampling_freq)
            data_chunk = data_df[:duration_in_samples].copy()

        data_chunk['time'] = data_chunk.index / sampling_freq
        data_plot = data_chunk.plot(
            subplots=True,
            width=300,
            x='time',
            xlabel='Time (s)',
            ylabel='Amplitude',
            shared_axes=True

        )

        status = r.hget(name='signal:'+filename, key='status')
        status_panel = pn.pane.Str(f"Status: {status.decode('utf-8')}")

        row = pn.Row(data_plot, status_panel)
        return row


layout = pn.Column()
waveform1 = Waveform()
waveform2 = ResampledWaveform()

layout.append(pn.Row(waveform1.param, waveform1.view_waveform))
layout.append(pn.Row(waveform2.param, waveform2.view_waveform))
