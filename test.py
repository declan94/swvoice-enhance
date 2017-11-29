from utils import ioutil, waveutil, plotutil
from ae.autoencoder import Autoencoder
import matplotlib.pyplot as plt
import tensorflow as tf

from configs import *

def test(path):
    x = ioutil.loadWaveFile(path)
    f, t, Zxx = waveutil.calcSTFT(x, window_len, frame_len)
    db, an = waveutil.stft2powerAngle(Zxx)
    autoencoder = Autoencoder(
        n_input=input_len,
        n_hidden=hidden_len,
        transfer_function=tf.nn.softplus,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001))
    autoencoder.loadModel("model/ae.ckpt")
    cnt = db.shape[1]/vector_frames
    in_data = db[:, :cnt*vector_frames].T.reshape(cnt, flen*vector_frames)
    out_data = autoencoder.reconstruct(in_data)
    out_db = out_data.reshape(cnt * vector_frames, flen).T
    plt.subplot(1, 2, 1)
    plotutil.plotTimeFreq(f, t, db, False)
    plt.subplot(1, 2, 2)
    plotutil.plotTimeFreq(f, t[:out_db.shape[1]], out_db)
    out_Zxx = waveutil.powerAngle2STFT(out_db, an[:,:out_db.shape[1]])
    t, out_x = waveutil.calcISTFT(out_Zxx, window_len, frame_len)
    ioutil.saveWaveFile(out_x, 'test_out.wav')

if __name__ == '__main__':
    test('test.wav')