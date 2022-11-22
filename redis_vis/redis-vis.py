import os
import pickle

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param
import redis

pd.options.plotting.backend = "hvplot"

r = redis.Redis(host='localhost', port=6379)

FILES = []
for file in r.keys(pattern='signal:*'):
    FILES.append(int(file.decode('utf-8').split(':')[-1]))
FILES.sort()

class Data(param.Parameterized):
    files = param.Selector(FILES)

    @staticmethod
    def get_df(filename, duration_in_s=0.01):
        # get the data from redis and put it in a dataframe with columns
        # x, y, z and time
        x_axis = pickle.loads(r.hget(name="signal:"+str(filename), key='x'))
        y_axis = pickle.loads(r.hget(name="signal:"+str(filename), key='y'))
        z_axis = pickle.loads(r.hget(name="signal:"+str(filename), key='z'))

        sampling_freq = 2000
        signal = {
            'x' : x_axis,
            'y' : y_axis,
            'z' : z_axis
        }
        data_chunk = pd.DataFrame(signal)

        return data_chunk


class Waveform(Data):
    def view_waveform(self):
        data_chunk = self.get_df(self.files)
        sampling_freq = 2000
        data_chunk['time'] = data_chunk.index / sampling_freq
        data_plot = data_chunk.plot(
            subplots=True,
            width=300,
            x='time',
            xlabel='Time (s)',
            ylabel='Amplitude',
        )

        row = pn.Row(data_plot)
        return row

layout = pn.Column()
waveform1 = Waveform()
layout.append(pn.Row(waveform1.param, waveform1.view_waveform))
waveform2 = Waveform()
layout.append(pn.Row(waveform2.param, waveform2.view_waveform))
layout.servable()