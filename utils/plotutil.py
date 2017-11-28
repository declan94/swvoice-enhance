import matplotlib.pyplot as plt
import numpy as np

def plotTimeFreq(f, t, z):
    plt.pcolormesh(t, f, np.log10(np.abs(z)))
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()