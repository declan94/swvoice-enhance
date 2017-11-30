from utils import ioutil, waveutil
from ae.autoencoder import Autoencoder, StackedAE
from os.path import isfile
import numpy as np
import tensorflow as tf

from train import trainAE
from configs import *

def getRandomBlock(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

def train(train_set):
    n_in = input_len
    train_data = train_set
    i = 0
    aes = []
    for n_hid in hidden_lens:
        i = i + 1
        print("Start training autoencoder %d: (%d -> %d)" % (i, n_in, n_hid))
        ae = trainAE(train_data, n_in, n_hid, 80)
        aes.append(ae)
        n_in = n_hid
        train_data = ae.encode(train_data)
    
    dae = StackedAE(input_len, aes)
    print("Start fine tuning")
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
            cost = dae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / total_batch

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%d,' % (epoch + 1), "Cost:", "{:.2f}".format(avg_cost))

    return dae



def main(train_dir):
    train_set = ioutil.loadTrainSet(train_dir, window_len, frame_len, vector_frames)
    dae = train(train_set)
    dae.saveModel("model/dae.ckpt")


if __name__ == '__main__':
    main('trainwav')


