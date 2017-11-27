import scipy.io.wavfile as wavfile

def loadWaveFile(filepath):
	data = wavfile.read(filepath)
	if len(data) != 2:
		return None
	return data[1][:, 0]
