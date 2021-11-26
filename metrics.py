from pystoi.stoi import stoi


def calculate_stoi(deg, ref, sampling_rate=16000):
    deg = deg.squeeze()
    ref = ref.squeeze()
    value = stoi(ref, deg, sampling_rate, extended=False)
    if value < 0.:
        value = 0.
    elif value > 1.:
        value = 1.
    return value
