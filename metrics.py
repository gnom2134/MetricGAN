from pesq import pesq
from pystoi.stoi import stoi


def calculate_pesq(sig1, sig2, sampling_rate=16000):
    return pesq(sampling_rate, sig1, sig2, 'wb')


def calculate_stoi(sig1, sig2, sampling_rate=16000):
    return stoi(sig1, sig2, sampling_rate, extended=False)
