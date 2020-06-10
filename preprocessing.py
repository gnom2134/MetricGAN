import librosa
import scipy.signal
import numpy as np
import torch


def prepare_signal(signal):
    signal = signal.squeeze().cpu().numpy()
    n_fft = 512
    signal_length = signal.shape[0]
    signal = librosa.util.fix_length(signal, signal_length + n_fft // 2)
    temp = librosa.stft(signal, n_fft=n_fft, hop_length=n_fft // 2, win_length=n_fft, window=scipy.signal.hanning)
    ft = np.abs(temp)

    phase = np.angle(temp)
    ft_mean = np.mean(ft, axis=1).reshape((257, 1))
    ft_std = np.std(ft, axis=1).reshape((257, 1)) + 1e-12

    normalised_ft = (ft - ft_mean) / ft_std

    return torch.tensor(normalised_ft), phase, signal_length


def back_to_wav(mag, phase, signal_length):
    n_fft = 512
    rec = np.multiply(mag.detach().cpu().numpy(), np.exp(1j * phase).T)
    result = librosa.istft(rec.squeeze().T,
                           hop_length=n_fft // 2,
                           win_length=n_fft,
                           window=scipy.signal.hanning, length=signal_length)
    return result


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import librosa.display

    y, sp = librosa.load("data/lisa/data/timit/raw/TIMIT/TRAIN/DR1/FCJF0/SA1.WAV")
    librosa.display.specshow(librosa.amplitude_to_db(prepare_signal(y), ref=np.max))
    plt.title('Power spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()
