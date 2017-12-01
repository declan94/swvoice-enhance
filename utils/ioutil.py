import scipy.io.wavfile as wavfile
import cPickle
import waveutil
import numpy as np
import os
import os.path
from sklearn import preprocessing


def listWaveFiles(dirpath):
	paths = [os.path.join(dirpath, f) for f in os.listdir(dirpath)]
	wavpaths = [p for p in paths if os.path.isfile(p) and p[-4:] == '.wav']
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

def loadTrainSet(train_dir, window_len, frame_len, vector_frames, cache=None):
    if cache != None and os.path.isfile(cache):
        return loadData(cache)
    flen = window_len/2 + 1
    first = True
    for p in listWaveFiles(train_dir):
        x = loadWaveFile(p)
        _, _, Zxx = waveutil.calcSTFT(x, window_len, frame_len)
        db, _ = waveutil.stft2powerAngle(Zxx)
        cnt = db.shape[1]/vector_frames
        train_in = db[:, :cnt*vector_frames].T.reshape(cnt, flen*vector_frames)
        scaler = preprocessing.MinMaxScaler(copy=False)
        scaler.fit_transform(train_in)
        if first:
            train_set = train_in
            first = False
        else:
            train_set = np.concatenate((train_set, train_in), axis=0)
    if cache != None:
        saveData(train_set, cache)
    return train_set

def ensureDirectory(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
