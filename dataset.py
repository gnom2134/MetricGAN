import torch
import os
import librosa
import numpy as np


TEST_DATA_FOLDER = 'data/lisa/data/timit/raw/TIMIT/TEST/'
TRAIN_DATA_FOLDER = 'data/lisa/data/timit/raw/TIMIT/TRAIN/'
NOISE_DATA_FOLDER = 'data/Nonspeech/'


def add_noise(signal, noise, snr):
    rms_sound = np.sqrt(np.mean(signal ** 2))
    rms_noise_desired = rms_sound * pow(10, snr / 40)

    rms_noise_current = np.sqrt(np.mean(noise ** 2))
    noise = noise * (rms_noise_desired / rms_noise_current)

    if len(noise) < len(signal):
        while len(noise) < len(signal):
            noise = np.hstack((noise, noise))
    noise = noise[0:len(signal)]

    return signal + noise


class TIMITDataset:
    def __init__(self, folder, noise_folder):
        self.file_paths = []
        self.noise_paths = []
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith('.WAV'):
                    self.file_paths.append(os.path.join(root, file))

        for root, _, files in os.walk(noise_folder):
            for file in files:
                if file.endswith('.wav'):
                    self.noise_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        signal, _ = librosa.load(self.file_paths[index])
        noise, _ = librosa.load(np.random.choice(self.noise_paths))
        noisy_signal = add_noise(signal, noise, np.random.choice(np.arange(-10, 11, 1)))
        return noisy_signal, signal


class MixedDataset:
    def __init__(self, enhanced_signals, folder, noise_folder):
        """
        :param enhanced_signals: list with tuples of format (enhanced_signal, clear_signal).
        :param folder:
        :param noise_folder:
        """
        self.timit_dataset = TIMITDataset(folder, noise_folder)
        self.enhanced_signals = enhanced_signals

    def __len__(self):
        return len(self.timit_dataset) + len(self.enhanced_signals)

    def __getitem__(self, index):
        if index >= len(self.timit_dataset):
            index -= len(self.timit_dataset)
            return self.enhanced_signals[index]
        else:
            return self.timit_dataset[index][1], self.timit_dataset[index][1]


if __name__ == "__main__":
    sig, sample_rate = librosa.load("data/lisa/data/timit/raw/TIMIT/TRAIN/DR1/FCJF0/SA1.WAV", sr=16000)
    n, _ = librosa.load("data/Nonspeech/n1.wav", sr=16000)

    res = add_noise(sig, n, 0)

    from scipy.io.wavfile import write
    write("data/results/res1_real.wav", sample_rate, sig)
    write("data/results/res1_noisy.wav", sample_rate, res)
