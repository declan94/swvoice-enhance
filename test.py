from utils import ioutil, waveutil, plotutil
from ae.autoencoder import Autoencoder
from sklearn import preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import os.path

from configs import *

def test(path, outpath):
    x = ioutil.loadWaveFile(path)
    f, t, Zxx = waveutil.calcSTFT(x, window_len, frame_len)
    db, an = waveutil.stft2powerAngle(Zxx)
    autoencoder = Autoencoder(
        n_input=input_len,
        n_hidden=hidden_len,
        transfer_function=tf.nn.softplus,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001))
    autoencoder.load_model("model/ae")
    cnt = db.shape[1]/vector_frames
    in_data = db[:, :cnt*vector_frames].T.reshape(cnt, flen*vector_frames)
    scaler = preprocessing.MinMaxScaler()
    in_data = scaler.fit_transform(in_data)
    out_data = autoencoder.reconstruct(in_data)
    out_data = scaler.inverse_transform(out_data)
    out_db = out_data.reshape(cnt * vector_frames, flen).T
    out_Zxx = waveutil.powerAngle2STFT(out_db, an[:,:out_db.shape[1]])
    out_t, out_x = waveutil.calcISTFT(out_Zxx, window_len, frame_len)
    ioutil.saveWaveFile(out_x, outpath)
    plt.subplot(1, 2, 1)
    plotutil.plotTimeFreq(f, t, db, False)
    plt.subplot(1, 2, 2)
    plotutil.plotTimeFreq(f, t[:out_db.shape[1]], out_db)

if __name__ == '__main__':
    inputfile = "test.wav"
    outputfile = None
    if len(sys.argv) > 1:
        inputfile = sys.argv[1]
    if len(sys.argv) > 2:
        outputfile = sys.argv[2]
    if not os.path.isfile(inputfile):
        inputfile = os.path.join("testwav", inputfile)
    if outputfile == None:
        outputfile = inputfile[:-4] + "_out.wav"
    test(inputfile, outputfile)

