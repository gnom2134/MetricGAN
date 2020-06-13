import librosa
import scipy.signal
import numpy as np
import torch


def prepare_signal(signal, normalize=False):
    signal = signal.squeeze().cpu().numpy()
    n_fft = 512
    signal_length = signal.shape[0]
    signal = librosa.util.fix_length(signal, signal_length + n_fft // 2)
    temp = librosa.stft(signal, n_fft=n_fft, hop_length=n_fft // 2, win_length=n_fft, window=scipy.signal.hanning)
    ft = np.abs(temp)

    phase = np.angle(temp)
    if normalize:
        ft_mean = np.mean(ft, axis=1).reshape((257, 1))
        ft_std = np.std(ft, axis=1).reshape((257, 1)) + 1e-12

        ft = (ft - ft_mean) / ft_std

    return torch.tensor(ft), phase, signal_length


def back_to_wav(mag, phase, signal_length):
    n_fft = 512
    rec = np.multiply(mag.detach().cpu().numpy(), np.exp(1j * phase))
    result = librosa.istft(rec.squeeze(),
                           hop_length=n_fft // 2,
                           win_length=n_fft,
                           window=scipy.signal.hanning, length=signal_length)
    return result


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import librosa.display

    y, _ = librosa.load("data/lisa/data/timit/raw/TIMIT/TRAIN/DR1/FCJF0/SA1.WAV", sr=16000)
    from scipy.io.wavfile import write

    write("data/results/res1_real.wav", 16000, y)
    write("data/results/res1_fixed.wav", 16000, back_to_wav(*prepare_signal(torch.tensor(y.squeeze()))))
    # librosa.display.specshow(librosa.amplitude_to_db(prepare_signal(y), ref=np.max))
    # plt.title('Power spectrogram')
    # plt.colorbar(format='%+2.0f dB')
    # plt.tight_layout()
    # plt.show()
