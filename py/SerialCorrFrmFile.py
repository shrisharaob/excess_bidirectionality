import numpy as np
import code
import sys
import pylab as plt
import scipy.stats as stat
from multiprocessing import Pool
from functools import partial
basefolder = "/homecentral/srao/Documents/code/mypybox"
sys.path.append(basefolder + "/nda/spkStats")
sys.path.append(basefolder + "/utils")
from DefaultArgs import DefaultArgs
from reportfig import ReportFig
from Print2Pdf import Print2Pdf

def SerialCor(spkTimes):
    out = (0.0, np.nan)
    if(spkTimes.size > 20):
       isi = np.diff(spkTimes)
       out =  stat.spearmanr(isi[0:-1], isi[1:]) # return 1st-order spearman rank correrelation coefficient
    return out

def SerialCorDistr(spkArray, neuronsList):
    sc = np.zeros((neuronsList.size, ))
    pVal = np.zeros((neuronsList.size, ))
    out = np.zeros((neuronsList.size, 3))
    alpha = 0.05
    for k, kNeuron in enumerate(neuronsList):
        spkTimes = spkArray[spkArray[:, 1] == kNeuron, 0]
        tmp = SerialCor(spkTimes)
        sc[k] = tmp[0]
  #      fr[k] = float(nSpks) / simDuration * 1e-3
        pVal[k] = tmp[1]
        out[k, 0] = tmp[0]
        out[k, 1] = tmp[1]
#        out[k, :] = np.array([tmp[0], tmp[1], float(nSpks) / simDuration * 1e-3])        
    return (sc, pVal)

if __name__ == '__main__':
    [foldername, alpha, filetag, NE, NI, xi, nTheta, contrast, tau_syn, simDuration, nTrials, dt] = DefaultArgs(sys.argv[1:], ['', '', '', 20000, 20000, '0.8', 8, 100.0, 3.0, 100000, 1, 0.05])

    bf = '/homecentral/srao/cuda/data/poster/'
    try:
        st = np.loadtxt(bf + foldername, delimiter = ';')
    except IOError:
        st = np.load(bf + foldername)
    except ValueError:
        st = np.load(bf + foldername)        
    
    pfunc = partial(SerialCorDistr, st)
    #p = Pool(8)
    #serialCorr = p.map(pfunc, np.arange(NE+NI))
    serialCorr = pfunc(np.arange(NE+NI))
#    serialCorr = pfunc(np.arange(10))    
    np.save('serialCorr_' + filetag, serialCorr)
