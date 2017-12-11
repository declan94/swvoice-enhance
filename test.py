
from utils import ioutil, waveutil, plotutil
from ae.autoencoder import StackedAE, Autoencoder
from sklearn import preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import os.path
import os
import numpy as np

from configs import *


def rebuildModel():
    autoencoder = Autoencoder(
        n_input=input_len,
        n_hidden=hidden_len,
        transfer_function=tf.nn.softplus,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001))
    autoencoder.load_model("model/ae")
    return autoencoder

def test(path, outpath):
    autoencoder = rebuildModel()
    scaler = ioutil.loadData("model/ae/scaler.pkl")
    scaler.set_params(copy=True)
    test_data = []
    for p in ioutil.listWaveFiles(path):
        name = os.path.basename(p)
        mos = float(name.split("-", 1)[0])
        x = ioutil.loadWaveFile(p)
        f, t, Zxx = waveutil.calcSTFT(x, window_len, frame_len)
        f, mels = waveutil.melFilter(f, Zxx, flen)
        cnt = mels.shape[1]/vector_frames
        test_in = mels[:, :cnt*vector_frames].T.reshape(cnt, flen*vector_frames)
        test_in = scaler.transform(test_in)
        test_out = autoencoder.reconstruct(test_in)
        mels_out = scaler.inverse_transform(test_out)
        mels_out = mels_out.reshape(cnt * vector_frames, flen).T

        t = t[:mels_out.shape[1]]
        mels = mels[:, :mels_out.shape[1]]
        
        cents = [70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90]
        mels_bs = [mels > np.percentile(mels, cent) for cent in cents]
        mels_out_bs = [mels_out > np.percentile(mels_out, cent) for cent in cents]
        diff_imgs = [np.logical_xor(mels_bs[i], mels_out_bs[i]) for i in range(0, len(cents))]
        diffs = [float(np.sum(diff_imgs[i])) / (100-cents[i]) for i in range(0, len(cents))]
        diff = min(diffs)
        i = diffs.index(diff)
        
        # diff = np.sum(diff_img)
        # diff = np.mean(np.abs(test_out - test_in))
        
        test_data.append({'path': p, 'mos': mos, 'diff': diff})
        print(mos, diff)
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plotutil.plotTimeFreq(f, t, mels)
        plt.subplot(1, 3, 2)
        plotutil.plotTimeFreq(f, t, mels_out)
        plt.subplot(1, 3, 3)
        plotutil.plotTimeFreq(f, t, abs(mels-mels_out))
        plt.savefig(os.path.join(outpath, name[:-4] + ".jpg"))
        plt.close()

    ioutil.saveData(test_data, os.path.join(outpath, "data.pkl"))
    mos = np.array([d['mos'] for d in test_data])
    diff = np.array([d['diff'] for d in test_data])
    z = np.polyfit(diff, mos, 1)
    p = np.poly1d(z)
    pred_mos = p(diff)
    mmos = np.mean(mos)
    ms = np.mean(diff)
    rho = -np.sum((mos - mmos) * (diff - ms)) / np.sqrt( np.sum((mos - mmos) ** 2) * np.sum((diff - ms) ** 2))
    print "rho: ", rho
    plt.plot(diff, mos, 'b.')
    plt.plot(diff, pred_mos, 'r.')
    plt.savefig(os.path.join(outpath, 'regress.jpg'))


    

if __name__ == '__main__':
    indir = "mos-audio"
    outdir = "mos-test-out"
    if len(sys.argv) > 1:
        indir = sys.argv[1]
    if len(sys.argv) > 2:
        outdir = sys.argv[2]
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    test(indir, outdir)



