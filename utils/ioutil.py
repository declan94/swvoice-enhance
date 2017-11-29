import scipy.io.wavfile as wavfile
import cPickle
from os import listdir
from os.path import isfile, join

def listWaveFiles(dirpath):
	paths = [join(dirpath, f) for f in listdir(dirpath)]
	wavpaths = [p for p in paths if isfile(p) and p[-4:] == '.wav']
	return wavpaths

def loadWaveFile(filepath):
    sr, wav = wavfile.read(filepath)
    assert(sr == 8000)
    if wav.ndim > 1:
        return wav[:, 0]
    else:
        return wav

def saveWaveFile(x, filepath):
	wavfile.write(filepath, 8000, x)

def saveData(obj, filepath):
	with open(filepath, 'wb') as output:
		cPickle.dump(obj, output)

def loadData(filepath):
	with open(filepath, 'rb') as inputPkl:
		return cPickle.load(inputPkl)
