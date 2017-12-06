from utils import ioutil, waveutil, plotutil

from configs import *

if __name__ == '__main__':
	x = ioutil.loadWaveFile('mos-audio/4.6250-1_17-12_19-17655-B-C-1-slice-1.wav')
	f, t, Zxx = waveutil.calcSTFT(x, window_len, frame_len)
	f, mels = waveutil.melFilter(f, Zxx, flen)
	plotutil.plotTimeFreq(f, t, mels, True)