from utils import ioutil, waveutil, plotutil
from ae.autoencoder import StackedAE, Autoencoder
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import os.path

from configs import *

def test(path, outpath):
    x = ioutil.loadWaveFile(path)
    f, t, Zxx = waveutil.calcSTFT(x, window_len, frame_len)
    db, an = waveutil.stft2powerAngle(Zxx)
    aes = []
    n_in = input_len
    for hid in hidden_lens:
        aes.append(Autoencoder(n_in, hid))
        n_in = hid
    autoencoder = StackedAE(input_len, aes)
    autoencoder.loadModel("model/dae.ckpt")
    cnt = db.shape[1]/vector_frames
    in_data = db[:, :cnt*vector_frames].T.reshape(cnt, flen*vector_frames)
    out_data = autoencoder.reconstruct(in_data)
    out_db = out_data.reshape(cnt * vector_frames, flen).T
    out_Zxx = waveutil.powerAngle2STFT(out_db, an[:,:out_db.shape[1]])
    # out_Zxx = Zxx
    out_t, out_x = waveutil.calcISTFT(out_Zxx, window_len, frame_len)
    ioutil.saveWaveFile(out_x, outpath)
    plt.subplot(1, 2, 1)
    plotutil.plotTimeFreq(f, t, db, False)
    plt.subplot(1, 2, 2)
    plotutil.plotTimeFreq(f, t[:out_db.shape[1]], out_db)

if __name__ == '__main__':
    print sys.argv
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