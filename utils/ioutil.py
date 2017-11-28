import scipy.io.wavfile as wavfile

def loadWaveFile(filepath):
    sr, wav = wavfile.read(filepath)
    assert(sr == 8000)
    if wav.ndim > 1:
        return wav[:, 0]
    else:
        return wav
