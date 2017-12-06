
# common
window_len = 256
frame_len = 64
# flen = window_len/2 + 1
flen = 40
vector_frames = 7
input_len = vector_frames * flen

# simple autoencoder
hidden_len = 80
train_epochs = 100

# stacked deep autoencoder
hidden_lens = [400, 100, 20]
dae_train_epochs = [100, 50, 50, 100]