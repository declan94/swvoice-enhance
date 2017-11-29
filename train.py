from utils import ioutil, waveutil
from ae.autoencoder import Autoencoder
from os.path import isfile
import numpy as np
import tensorflow as tf

from configs import *

def loadTrainSet(train_dir, cache=None):
    if cache != None and isfile(cache):
        return ioutil.loadData(cache)
    first = True
    for p in ioutil.listWaveFiles(train_dir):
        x = ioutil.loadWaveFile(p)
        _, _, Zxx = waveutil.calcSTFT(x, window_len, frame_len)
        db, _ = waveutil.stft2powerAngle(Zxx)
        cnt = db.shape[1]/vector_frames
        train_in = db[:, :cnt*vector_frames].T.reshape(cnt, flen*vector_frames)
        if first:
            train_set = train_in
            first = False
        else:
            train_set = np.concatenate((train_set, train_in), axis=0)
    if cache != None:
        ioutil.saveData(train_set, cache)
    return train_set

def getRandomBlock(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


def train(train_set):
    autoencoder = Autoencoder(
        n_input=input_len,
        n_hidden=hidden_len,
        transfer_function=tf.nn.softplus,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001))
    n_samples = train_set.shape[0]
    training_epochs = 80
    batch_size = 256
    display_step = 1

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = getRandomBlock(train_set, batch_size)
            # Fit training using batch data
            cost = autoencoder.partial_fit(batch_xs)
        # Compute average loss
        avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%d,' % (epoch + 1), "Cost:", "{:.9f}".format(avg_cost))

    return autoencoder



def main(train_dir):
    train_set = loadTrainSet(train_dir)
    ae = train(train_set)
    ae.saveModel("model/ae.ckpt")


if __name__ == '__main__':
    main('trainwav')


