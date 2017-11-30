
# common
window_len = 128
frame_len = 64
flen = window_len/2 + 1
vector_frames = 8
input_len = vector_frames * flen

# simple autoencoder
hidden_len = 300

# stacked deep autoencoder
hidden_lens = [600, 300, 100]