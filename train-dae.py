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
    ae_weights = []
    for n_hid in hidden_lens:
        print("Start training autoencoder %d: (%d -> %d)" % (i, n_in, n_hid))
        ae = trainAE(train_data, n_in, n_hid, dae_train_epochs[i], name="ae-{}".format(i))
        ae_weights.append(ae.get_weights())
        n_in = n_hid
        train_data = ae.encode(train_data)
        i = i + 1
    
    tf.reset_default_graph()
    dae = StackedAE(input_len, ae_weights)
    print("Start fine tuning")
    n_samples = train_set.shape[0]
    training_epochs = dae_train_epochs[-1]
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
            print("Epoch:", '%d,' % (epoch + 1), "Cost:", "{:.8f}".format(avg_cost))

        if epoch > 0 and epoch % 10 == 0:
            dae.save_model("model/dae/dae.ckpt")
            print("temp model saved")

    return dae



def main(train_dir):
    train_set = ioutil.loadTrainSetMel(train_dir, window_len, frame_len, vector_frames, flen, 'trainwav/traindata.pkl')
    dae = train(train_set)
    dae.save_model("model/dae/dae.ckpt")
    
    # with tf.Session() as sess:
    #     writer = tf.summary.FileWriter("logs/", sess.graph)


if __name__ == '__main__':
    main('trainwav')


