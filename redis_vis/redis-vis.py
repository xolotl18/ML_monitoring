# serve this python script with panel to display an interactive dashboard
# the waveform visualization retrieves the signal data from redis keys
# the fft visualization triggers a redis ai model to convert the signal,
# stored as a key, to a spectrum and then displays it

import holoviews as hv
import ml2rt
import numpy as np
import pandas as pd
import panel as pn
import param
import redis
import redisai as rai
import torch
import torchaudio.transforms as tat

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

        signal = {
            'x': signal_x,
            'y': signal_y,
            'z': signal_z
        }
        data_chunk = pd.DataFrame(signal)

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
            shared_axes=True
        )
        data_panel = pn.pane.HoloViews(data_plot, linked_axes=False)

        return data_panel

class Spectrogram(Data):
    new_freq = param.Integer(2000, label='Resampling Frequency (Hz)', bounds=(500, 2000), step=100)
    win_length_in_ms = param.Integer(32, label='Win Length', bounds=(8, 128), step=8)
    hop_length_in_ms = param.Integer(16, label='Hop Length', bounds=(4, 64), step=4)
    n_mels = param.Integer(16, label='Bins', bounds=(4, 64), step=4)
    f_min = param.Integer(50, label='Fmin', bounds=(25, 250), step=25)
    f_max = param.Integer(1000, label='Fmax', bounds=(250, 1000), step=25)

    def view_spectrogram(self):
        filename = self.files
        sampling_freq = int(r.hget(name='signal:'+str(filename),
                            key='sampling_f'))
        signal_x = np.frombuffer(r.hget(name="signal:"+str(filename), key='x'),
                               dtype=np.float32)
        signal_y = np.frombuffer(r.hget(name="signal:"+str(filename), key='y'),
                               dtype=np.float32)
        signal_z = np.frombuffer(r.hget(name="signal:"+str(filename), key='z'),
                               dtype=np.float32)

        signal = {
            'x': signal_x,
            'y': signal_y,
            'z': signal_z
        }

        orig_freq = sampling_freq
        transforms_list = []

        if self.new_freq != orig_freq:
            transforms_list.append(tat.Resample(orig_freq, self.new_freq))

        win_length = int(self.win_length_in_ms * self.new_freq / 1000)
        hop_length = int(self.hop_length_in_ms * self.new_freq / 1000)
        transforms_list.append(
            tat.MelSpectrogram(
                sample_rate=self.new_freq,
                n_fft=win_length,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=self.n_mels,
                f_min=self.f_min,
                f_max=self.f_max,
                center=False,
            )
        )
        transforms_list.append(tat.AmplitudeToDB())

        transforms = torch.nn.Sequential(*transforms_list)

        # load the transform model to redisai and run it to obtain the output
        # spectrogram

        transform_model = torch.jit.script(transforms)
        transform_model.save('model.pt')
        binary_model = ml2rt.load_model('model.pt')

        con.modelstore('model:spectrogram', 'torch', 'cpu', data=binary_model)

        outx = con.tensorset('tensor:'+str(filename)+':x', signal_x)
        outy = con.tensorset('tensor:'+str(filename)+':y', signal_y)
        outz = con.tensorset('tensor:'+str(filename)+':z', signal_z)
        outx1 = con.modelrun('model:spectrogram', 'tensor:'+str(filename)+':x',
                             'spectrogram:'+str(filename)+':x')
        outy1 = con.modelrun('model:spectrogram', 'tensor:'+str(filename)+':y',
                             'spectrogram:'+str(filename)+':y')
        outz1 = con.modelrun('model:spectrogram', 'tensor:'+str(filename)+':z',
                             'spectrogram:'+str(filename)+':z')

        spectrogram_x = con.tensorget('spectrogram:'+str(filename)+':x')
        spectrogram_y = con.tensorget('spectrogram:'+str(filename)+':y')
        spectrogram_z = con.tensorget('spectrogram:'+str(filename)+':z')

        mel_spectrogram = {
            'x': spectrogram_x,
            'y': spectrogram_y,
            'z': spectrogram_z
        }

        plots = []

        mel_f_min = 2595.0 * np.log10(1.0 + (self.f_min / 700.0))
        mel_f_max = 2595.0 * np.log10(1.0 + (self.f_max / 700.0))
        mel_pts = np.linspace(mel_f_min, mel_f_max, self.n_mels)

        frequencies = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
        yticks = []
        for i, m in enumerate(frequencies):
            yticks.append((i, f'{m:.0f}'))

        for c in mel_spectrogram.keys():
            mel = mel_spectrogram[c]
            plot = hv.Image(
                mel,
                bounds=(0, 0, mel.shape[1], mel.shape[0]),
                vdims=hv.Dimension('z', range=(-8, 80)),
            ).opts(cmap='Viridis', yticks=yticks)
            plots.append(plot)

        layout = hv.Layout(plots)

        return layout


layout = pn.Column()
waveform1 = Waveform()
layout.append(pn.Row(waveform1.param, waveform1.view_waveform))
waveform2 = Waveform()
layout.append(pn.Row(waveform2.param, waveform2.view_waveform))
fft1 = FFT()
layout.append(pn.Row(fft1.param, fft1.view_fft))
spectrogram1 = Spectrogram()
layout.append(pn.Row(spectrogram1.param, spectrogram1.view_spectrogram))
layout.servable()
