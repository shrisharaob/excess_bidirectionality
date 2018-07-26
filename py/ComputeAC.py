
#!/usr/bin/python

# ''' python ComputeAC.py 8 3 i2i 2000'''
import numpy as np
import code
import sys
import pylab as plt
from multiprocessing import Pool
from functools import partial
basefolder = "/homecentral/srao/Documents/code/mypybox"
sys.path.append(basefolder + "/nda/spkStats")
sys.path.append(basefolder + "/utils")
from DefaultArgs import DefaultArgs
from reportfig import ReportFig
from Print2Pdf import Print2Pdf
import time

sys.path.append("/homecentral/srao/Documents/code/mypybox")
sys.path.append("/homecentral/srao/Documents/code/mypybox/utils")

def AutoCorr(x, corrLength = "same"):
    # x : spike Times in ms
    N = len(x)
    nPointFFT = int(np.power(2, np.ceil(np.log2(len(x)))))
    fftX = np.fft.fft(x, nPointFFT)
    return np.abs(np.abs(np.fft.ifft(np.multiply(fftX, np.conj(fftX)))))

def AvgAutoCorrInInterval(starray, neuronsList, spkTimeStart, spkTimeEnd, dbname, simDT = 0.05, minSpks = 100, maxTimeLag = 100, NE = 20000, NI = 20000, fileTag = 'E', theta = 0):
    N = len(neuronsList)
    pcDone = 0
    simDuration = spkTimeEnd - spkTimeStart
    nTimeLagBins = int(2 * maxTimeLag) # works if downsample bin size = 1ms otherwise divide by value
    avgCorr = np.zeros((int(np.power(2, np.ceil(np.log2(simDuration + simDT)))), ))
    binFlag = 0;
    nValidNeurons = 0;
    downSampleBinSize = 1
    spkBins = np.arange(spkTimeStart, spkTimeEnd + simDT, downSampleBinSize)
    nSpkBins = len(spkBins) ;
    avgRate = 0
    for i, kNeuron in enumerate(neuronsList):
        spkTimes = starray[starray[:, 1] == kNeuron, 0]
        spksTimes = spkTimes[spkTimes > spkTimeStart]
        meanRate = float(spkTimes.size) / float(simDuration * 1e-3)
        nValidNeurons += 1
        avgRate += meanRate
        if(spkTimes.size > minSpks):
            st = np.histogram(np.squeeze(spkTimes), spkBins)
            tmpCorr = AutoCorr(st[0])
           # avgCorr += tmpCorr / ((downSampleBinSize * 1e-3) **2 * nSpkBins * meanRate)
            avgCorr += tmpCorr / ((downSampleBinSize * 1e-3) **2 * nSpkBins)            
    avgCorr = avgCorr / nValidNeurons
    bins = np.array(downSampleBinSize)
    if(len(avgCorr) > 0):
        return avgCorr
    else :
        return 0

if __name__ == '__main__':
#    [foldername, alpha, tau_syn, bidirType, tag, NE, NI, xi, nTheta, contrast,  simDuration, nTrials, dt] = DefaultArgs(sys.argv[1:], ['', '', 3.0, '',  '', 20000, 20000, '0.8', 8, 100.0,  100000, 1, 0.05])
    [alpha, tau_syn, bidirType, K, NE, NI, tag, xi, nTheta, contrast,  simDuration, nTrials, dt] = DefaultArgs(sys.argv[1:], ['', 3.0, '',  '', 10000, 10000, '', '0.8', 8, 100.0, 20000, 1, 0.05])
    foldername = ''
    print 'loading st from file ...'
    NE = int(NE)
    NI = int(NI)
    K = int(K)

    tau_syn = int(tau_syn)
    simDuration = float(simDuration)
    spkTimeStart = 3000.0
    spkTimeEnd = simDuration
#    simDuration = spkTimeEnd - spkTimeStart
    sys.stdout.flush()
    # filebase = '/homecentral/srao/Documents/code/cuda/cudanw/data/pub/bidir/'
    # filebase = '/homecentral/srao/cuda/data/pub/bidir/i2i/tau3/p%s/K%s/'%(alpha, int(K))
    filebase = '/homecentral/srao/cuda/data/pub/bidir/N10000/KFFI/i2i/tau3/p%s/K%s/'%(alpha, int(K))    

    if tag == '':
        if NI == 20000:
            tag = 'NI2E4'
        elif NI == 50000:
            tag = 'NI5E4'
        elif NI == 60000:
            tag = 'NI6E4'
        elif NI == 10000:
            tag = 'NI1E4'
            # filebase = '/homecentral/srao/Documents/code/cuda/cudanw/data/pub/N1E4/'
            # if tau_syn == 3:
            #     foldername = 'i2i/p%s/'%(alpha)
            # else:
            #     foldername = 'i2i/p%s/tau%s/'%(alpha, tau_syn)
        elif NI == 30000:
            tag = 'NI3E4'
            filebase = '/homecentral/srao/Documents/code/cuda/cudanw/data/pub/N3E4/'
            foldername = 'tau%s/p%s/'%(tau_syn, alpha)
        elif NI == 40000:
            tag = 'NI4E4'
            filebase = '/homecentral/srao/Documents/code/cuda/cudanw/data/pub/N4E4/'
            foldername = 'tau%s/p%s/'%(tau_syn, alpha)            
            
    elif tag == 'NI1E4':
        NI = 10000
        filebase = '/homecentral/srao/Documents/code/cuda/cudanw/data/pub/N1E4/'
        if tau_syn == 3:
            foldername = 'i2i/p%s/'%(alpha)
        else:
            foldername = 'i2i/p%s/tau%s/'%(alpha, tau_syn)

    elif tag == 'NI2E4':
        NI = 20000
    elif tag == 'NI5E4':
        NI = 50000
    elif tag == 'NI6E4':
        NI = 10000
    maxLag = 2000
    bins = np.arange(-1000, 1000, 1)
    acMat = np.zeros((maxLag, 2, nTrials))
    print 'FILEBASE: ', filebase, foldername
    if int(tau_syn) == 48 and int(alpha) == 0:
        simDuration = 250000
        print simDuration
    for kTrial in range(nTrials):
        stfilename = 'spkTimes_xi0.8_theta0_0.%s_%s.0_cntrst100.0_%s_tr%s.csv'%(int(alpha), int(tau_syn), int(simDuration), 0)        
        stfilename_alternate = 'spkTimes_xi0.8_theta0_0.%s0_%s.0_cntrst100.0_%s_tr%s.csv'%(int(alpha), int(tau_syn), int(simDuration), 0)
        if K == 500 and NE == 20000:
            stfilename = 'spkTimes_xi0.8_theta0_0.%s_%s.0_cntrst100.0_%s_tr%s.csv'%(int(alpha), int(tau_syn), int(simDuration), 0)
            stfilename_alternate = 'spkTimes_xi0.8_theta0_0.%s0_%s.0_cntrst100.0_%s_tr%s.csv'%(int(alpha), int(tau_syn), int(simDuration), 0)
        print 'looking for file: ', stfilename, '\n', 'in folder:'
        fr = np.loadtxt(filebase + 'firingrates_xi0.8_theta%s_0.%s0_3.0_cntrst100.0_%s_tr0.csv'%(int(0), int(alpha), int(simDuration) ))
        print 'mean rates: ', fr[:NE].mean(), fr[NE:].mean()
        
        try:
            print foldername
            starray = np.loadtxt(filebase + foldername + stfilename, delimiter = ';')
            print 'SUCCESS!'

        except IOError:
            print "doest not exist!"
            print "loading file: ", stfilename_alternate, '...', 
            sys.stdout.flush()        
            try:
                starray = np.loadtxt(filebase + foldername + stfilename_alternate, delimiter = ';')
                print 'SUCCESS!'
            except IOError:
                try:
                    print 'searching in absolute path: ', foldername, '...', 
                    sys.stdout.flush()        
                    starray = np.loadtxt(foldername + stfilename, delimiter = ';')
                    print 'SUCCESS!'                    
                except IOError:
                    print "doest not exist!"
                    print "loading file: ", stfilename_alternate, '...',
                    sys.stdout.flush()        
                    try: 
                        starray = np.loadtxt(filebase + foldername + stfilename_alternate, delimiter = ';')
                        print 'SUCCESS!'
                    except IOError:
                        print "file not found!"
                        raise
        print 'file loaded!'
        # ---------- COMPUTE ---------------------
#        filetag = 'bidir_' + tag + '_bidirI2I_tau%s_p%s'%(tau_syn, int(alpha))
        tag = tag + '_K%s'%(K)
        filetag = 'bidir' + tag + '_tau%s_p%s'%(tau_syn, int(alpha))    
        listOfneurons = np.arange(NE)
    #        listOfneurons = np.unique(np.random.randint(0, NE, 1100)) #np.arange(NE)
        print 'computing ac E...',
        sys.stdout.flush()
        start_time = time.time()
        ac = AvgAutoCorrInInterval(starray, listOfneurons, spkTimeStart, spkTimeEnd, '', dt, 0, maxLag, NE, NI, 'EI')
        print 'done!'
        print("--- %s min ---" % ((time.time() - start_time) / 60))
        ac[np.argmax(ac)] = 0.0
#        acMat[:, 0] = acMat[:, 0] + ac[:maxLag]
        acMat[:, 0, kTrial] = ac[:maxLag]
        listOfneurons = np.arange(NE, NE+NI, 1)
#        listOfneurons = np.unique(np.random.randint(NE, NE + NI, 100)) #np.arange(NE)    
        print 'computing ac I ...',
        sys.stdout.flush()
        start_time = time.time()        
        ac = AvgAutoCorrInInterval(starray, listOfneurons, spkTimeStart, spkTimeEnd, '', dt, 0, maxLag, NE, NI, 'EI')
        print 'done!'
        print("--- %s min ---" % ((time.time() - start_time) / 60))        
        ac[np.argmax(ac)] = 0.0
 #       acMat[:, 1] = acMat[:, 1] + ac[:maxLag]
        acMat[:, 1, kTrial] = ac[:maxLag]        
#    acMat = acMat / nTrials
    saveFolder = './data/newAC_2018/' #/homecentral/srao/Documents/code/tmp/del/auto2/onepop/'
    saveFilename = 'long_tau_vs_ac_mat_tr%s_'%(nTrials, ) + filetag
    print "saving as: ", saveFilename
    np.save(saveFolder + saveFilename, acMat)
