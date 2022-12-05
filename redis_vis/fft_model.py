import torch


class FFT_compute(torch.nn.Module):
    def forward(self, x):
        spectrum = torch.fft.rfft(x)
        spectrum = torch.abs(spectrum)
        spectrum = torch.log(spectrum)
        return spectrum


scripted_cell = torch.jit.script(FFT_compute())
print(scripted_cell)
scripted_cell.save('fft.pt')
