from scipy import signal

def calcTimeFreq(x, windowLen, frameLen):
	overlap = windowLen - frameLen
	return signal.stft(x, fs=8000, window="hamming", nperseg=windowLen, noverlap=overlap)