from pesq import pesq
from pystoi.stoi import stoi


def calculate_pesq(deg, ref, sampling_rate=16000, squeeze=True):
    deg = deg.squeeze()
    ref = ref.squeeze()
    if squeeze:
        value = (pesq(sampling_rate, ref, deg, 'wb') + 0.5) / 5
        upper_bound = 1.
        lower_bound = 0.
    else:
        value = pesq(sampling_rate, ref, deg, 'wb')
        upper_bound = 4.5
        lower_bound = -0.5
    if value < lower_bound:
        value = lower_bound
    elif value > upper_bound:
        value = upper_bound
    return value


def calculate_stoi(deg, ref, sampling_rate=16000):
    deg = deg.squeeze()
    ref = ref.squeeze()
    value = stoi(ref, deg, sampling_rate, extended=False)
    if value < 0.:
        value = 0.
    elif value > 1.:
        value = 1.
    return value
