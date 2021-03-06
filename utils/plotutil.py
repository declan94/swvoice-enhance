import matplotlib.pyplot as plt
import numpy as np

def plotTimeFreq(f, t, powerDB, show=False):
    plt.pcolormesh(t, f, powerDB)
    plt.colorbar()
    plt.title('Power [dB]')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    if show:
    	plt.show()

