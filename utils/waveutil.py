from scipy import signal
import numpy as np

def calcSTFT(x, windowLen, frameLen):
    overlap = windowLen - frameLen
    f, t, Zxx = signal.stft(x, fs=8000, window="hamming", nperseg=windowLen, noverlap=overlap)
    return (f, t, Zxx)

def stft2powerAngle(Zxx):
    power = np.abs(Zxx) ** 2
    power[power==0] = np.min(power[power>0])
    powerDB = 20 * np.log10(power)
    angle = np.angle(Zxx)
    return (powerDB, angle)

def powerAngle2STFT(powerDB, angle):
    power = 10 ** (powerDB/20)
    amp = np.sqrt(power)
    return amp * np.exp(1j * angle)

def calcISTFT(Zxx, windowLen, frameLen, dtype=np.int16):
    overlap = windowLen - frameLen
    t, x = signal.istft(Zxx, fs=8000, window="hamming", nperseg=windowLen, noverlap=overlap)
    x = dtype(x)
    return (t, x)


def melFilter(f, Zxx, nfilter=40):
    NFFT = (len(f)-1) * 2
    sample_rate = 8000
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + f[-1] / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilter + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)
    fbank = np.zeros((nfilter, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilter + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    pow_frames = ((1.0 / NFFT) * (np.abs(Zxx) ** 2))
    filter_banks = np.dot(fbank, pow_frames)
    minp = np.min(filter_banks[filter_banks>0])
    filter_banks = np.where(filter_banks == 0, minp, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    return hz_points[1:-1], filter_banks