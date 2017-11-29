import matplotlib.pyplot as plt
import numpy as np

def plotTimeFreq(f, t, powerDB, show=True):
    plt.pcolormesh(t, f, powerDB)
    plt.title('Power [dB]')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    if show:
    	plt.show()

