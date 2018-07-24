import numpy as np
import pylab as plt

def plot():
    ack500 = np.load('long_tau_vs_ac_mat_tr1_bidirNI2E4E2I_K500_tau3_p8.npy')
    ack2000 = np.load('onepop/long_tau_vs_ac_mat_tr1_bidirNI2E4I2I_K2000_tau3_p8.npy') 
    
    plt.plot(np.squeeze(ack500[:, 0, 0]), label = 'K = 500')
    plt.plot(np.squeeze(ack2000[:, 0, 0]), label = 'K = 2000')

    plt.figure()
    plt.plot(np.squeeze(ack500[:, 1, 0]), label = 'K = 500')
    plt.plot(np.squeeze(ack2000[:, 1, 0]), label = 'K = 2000')
    
    plt.ion()
    plt.show()

    

