from scipy import signal
import numpy

def calcTimeFreq(x, windowLen, frameLen):
    overlap = windowLen - frameLen
    return signal.stft(x, fs=8000, window="hamming", nperseg=windowLen, noverlap=overlap)


def melFilter(f, Zxx, nfilter=40):
    NFFT = (len(f)-1) * 2
    sample_rate = 8000
    low_freq_mel = 0
    high_freq_mel = (2595 * numpy.log10(1 + f[-1] / 700))  # Convert Hz to Mel
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilter + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)
    fbank = numpy.zeros((nfilter, int(numpy.floor(NFFT / 2 + 1))))
    for m in range(1, nfilter + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    pow_frames = ((1.0 / NFFT) * (numpy.abs(Zxx) ** 2))
    filter_banks = numpy.dot(fbank, pow_frames)
    filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * numpy.log10(filter_banks)  # dB
    return hz_points[1:-1], filter_banks