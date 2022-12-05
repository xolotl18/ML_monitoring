import numpy as np
import pandas as pd
import panel as pn
import param
import redis
import redisai as rai

pd.options.plotting.backend = "hvplot"
debug = False

r = redis.Redis(host='localhost', port=6379)
con = rai.Client(host='localhost', port=6379)

FILES = []
for file in r.keys(pattern='signal:*'):
    FILES.append(int(file.decode('utf-8').split(':')[-1]))
FILES.sort()

processed_signals = [0 for i in np.arange(len(FILES))]


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


class FFT(Data):
    def view_fft(self):
        filename = self.files
        # data_chunk_waveform = self.get_df(filename)
        sampling_freq = int(r.hget(name='signal:'+str(filename),
                            key='sampling_f'))
        signal_x = np.frombuffer(r.hget(name="signal:"+str(filename), key='x'),
                                 dtype=np.float32)
        signal_y = np.frombuffer(r.hget(name="signal:"+str(filename), key='y'),
                                 dtype=np.float32)
        signal_z = np.frombuffer(r.hget(name="signal:"+str(filename), key='z'),
                                 dtype=np.float32)

        if processed_signals[int(filename)] == 0:
            outx = con.tensorset('tensor:'+str(filename)+':x', signal_x)
            outy = con.tensorset('tensor:'+str(filename)+':y', signal_y)
            outz = con.tensorset('tensor:'+str(filename)+':z', signal_z)
            outx1 = con.modelrun('model:fft', 'tensor:'+str(filename)+':x',
                                 'fft:'+str(filename)+':x')
            outy1 = con.modelrun('model:fft', 'tensor:'+str(filename)+':y',
                                 'fft:'+str(filename)+':y')
            outz1 = con.modelrun('model:fft', 'tensor:'+str(filename)+':z',
                                 'fft:'+str(filename)+':z')
            processed_signals[int(filename)] = 1
        if debug:
            print(outx)
            print(outy)
            print(outz)
            print(outx1)
            print(outy1)
            print(outz1)

        fft_x = con.tensorget('fft:'+str(filename)+':x')
        fft_y = con.tensorget('fft:'+str(filename)+':y')
        fft_z = con.tensorget('fft:'+str(filename)+':z')
        fft_all = {
            'x': fft_x,
            'y': fft_y,
            'z': fft_z
        }
        data_chunk = pd.DataFrame(fft_all)
        data_chunk['frequency'] = np.linspace(
            0, sampling_freq // 2, len(data_chunk)
        )
        data_plot = data_chunk.plot(
            subplots=True,
            width=300,
            x='frequency',
            xlabel='Frequency (Hz)',
            ylabel='Magnitude',
        )
        data_panel = pn.pane.HoloViews(data_plot, linked_axes=False,
                                       shared_axes=True)

        return data_panel


layout = pn.Column()
waveform1 = Waveform()
layout.append(pn.Row(waveform1.param, waveform1.view_waveform))
waveform2 = Waveform()
layout.append(pn.Row(waveform2.param, waveform2.view_waveform))
fft1 = FFT()
layout.append(pn.Row(fft1.param, fft1.view_fft))
layout.servable()
