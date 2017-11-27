from scipy import signal

def calcTimeFreq(x, windowLen, frameLen):
	overlap = windowLen - frameLen
	return signal.stft(x, 'hamming', windowLen, overlap)