from utils import ioutil, waveutil
from ae.autoencoder import Autoencoder
from os.path import isfile
from sklearn import preprocessing
import numpy as np
import tensorflow as tf


from configs import *

def getRandomBlock(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

def trainAE(train_set, n_in, n_hid, training_epochs = 80, save_interval = 10, name = "autoencoder"):
    autoencoder = Autoencoder(
        n_input=n_in,
        n_hidden=n_hid,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
        name = name)
    n_samples = train_set.shape[0]
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
            avg_cost += cost / total_batch

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%d,' % (epoch + 1), "Cost:", "{:.8f}".format(avg_cost))

        if save_interval > 0 and epoch > 0 and epoch % save_interval == 0:
            autoencoder.save_model("model/ae")
            print("temp model saved")


    return autoencoder



def main(train_dir):
    train_set, scaler = ioutil.loadTrainSetMel(train_dir, window_len, frame_len, vector_frames, flen)
    ioutil.saveData(scaler, "model/ae/scaler.pkl")
    ae = trainAE(train_set, input_len, hidden_len, train_epochs, 10)
    ae.save_model("model/ae")
    # with tf.Session() as sess:
    #     writer = tf.summary.FileWriter("logs/", sess.graph)


if __name__ == '__main__':
    main('trainwav')


