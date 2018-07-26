basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import pylab as plt
from scipy.optimize import curve_fit
import sys, os
sys.path.append(basefolder)
sys.path.append(basefolder + "/utils")
import Print2Pdf as pdf
import Keyboard as kb

pPossible = [0, 1, 2, 3, 4, 5, 6, 7, 75, 8, 82, 84, 85, 86, 88, 9, 95, 97]
realpPossible = [0, .1, .2, .3, .4, .5, .6, .7, .75, .8, .82, .84, .85, .86, .88, .9, .95, .97]



#p = [1, 2, 3, 4, 5, 6, 7, 75, 8, 9]
#realp = [.1, .2, .3, .4, .5, .6, .7, .75, .8, .9]

# p =     [5, 7,  8, 9, 95,  97]
# plotP = [.5, .7,  .8, .9, .95, .97]

tau_s = [3, 6, 12, 24, 48]
#tau_s = [48]
neuronType = int(sys.argv[1]) # E = 0, I = 1
neuronTag = 'E'
figFolder = '/homecentral/srao/Documents/code/tmp/figs/publication_figures/powerlaw/'
if neuronType == 1:
    neuronTag = 'I'
print "neuron type: ", neuronTag
dataFolder = './'
validStartIdxCnst = 20
#paperSize = [2.0*1.5, 2.0]
#taudecayFiletag = sys.argv[1]
estTau_dec_list = []
estTauError_dec_list = []
p_list = []
paperSize = [2.0, 1.5]
axPosition = [0.26, 0.25, .65, 0.65]
plt.ioff()

def func(x, a, tau, c):
    return a*np.exp(- x / tau) + c

def RemoveIdicesFromList(somelist, indices2Remove):
    for idx in sorted(indices2Remove, reverse = True):
        del somelist[idx]
    return somelist


if __name__ == '__main__':
    datafolder = '/homecentral/srao/Documents/code/tmp/data/'
    for kk, kTau in enumerate(tau_s):
        pPossible = [0, 1, 2, 3, 4, 5, 6, 7, 75, 8, 82, 84, 85, 86, 88, 9, 95, 97]
        realpPossible = [0, .1, .2, .3, .4, .5, .6, .7, .75, .8, .82, .84, .85, .86, .88, .9, .95, .97]
        # pPossible = [0, 1, 2, 3, 4, 5, 6, 7]
        # realpPossible = [0, .1, .2, .3, .4, .5, .6, .7] 
        plt.ion()
        filenames = []
        if kTau == 3:
            filenames = [datafolder + 'long_tau_vs_ac_mat_tr1_' + 'bidirI2I_tau%s_p%s.npy'%((kTau, x[1])) for x in enumerate(pPossible)]
        else:
            filenames = [datafolder + 'long_tau_vs_ac_mat_tr1_' + 'bidirNI2E4I2I_tau%s_p%s.npy'%((kTau, x[1])) for x in enumerate(pPossible)]
#         if kTau == 3:
#             filenames = [datafolder + 'long_tau_vs_ac_mat_tr1_' + 'bidirI2I_tau3_p%s'%(x[1]) for x in enumerate(alphas)]
# #            filenames = ['long_tau_vs_ac_mat_bidir_I2I_p%s.npy'%(pp[1]) for pp in enumerate(p)]
#         elif kTau == 6 or kTau == 12:
#             filenames = ['long_tau_vs_ac_mat_bidir_bidirI2I_p%s_tau%s_p%s.npy'%(pp[1], kTau, pp[1]) for pp in enumerate(p)]
#         else:
#             filenames = ['long_tau_vs_ac_mat_bidir_bidirI2I_tau%s_p%s.npy'%(kTau, pp[1]) for pp in enumerate(p)]
#        filenames = ['long_tau_vs_ac_mat_bidir_%s_bidirI2I_tau%s_p%s.npy'%(taudecayFiletag, kTau, ip) for ip in p]
#        filenames = ['long_tau_vs_ac_mat_tr10_bidir_%s_bidirI2I_tau%s_p%s.npy'%(taudecayFiletag, kTau, ip) for ip in p]

        indices2Remove = []
        for ll, lFileName in enumerate(filenames):
            if not os.path.isfile(lFileName):
                indices2Remove.append(ll)                
        filenames = RemoveIdicesFromList(filenames, indices2Remove)
        p = RemoveIdicesFromList(pPossible, indices2Remove)
        realp = RemoveIdicesFromList(realpPossible, indices2Remove)
        p_list.append(realp)
        estTau_dec = np.zeros((len(p), ))
        estTau_error = np.zeros((len(p), ))
        for idxp, bidirp in enumerate(p):
            print 'loading file: ', filenames[idxp]
            ac = np.squeeze(np.load(filenames[idxp]))
            tmpMaxAC = np.argmax(ac[:, neuronType])
            print ac.shape
            if kTau == 3:
                validStartIdx = 10 #18 #min([validStartIdxCnst, tmpMaxAC])
            elif kTau == 48:
                validStartIdx = 22                
            else:
                validStartIdx = 15 #22 #min([validStartIdxCnst, tmpMaxAC]) 
            if ac.ndim == 2:
                y = ac[validStartIdx:, neuronType]
            elif ac.ndim == 3:
                y = ac[validStartIdx:, 1, :].mean(1)
                # tmp0 = ac
                # d0, d1, d2 = ac.shape
                # tmp1 = np.zeros((d0, d1, d2))
                # tmp1[:, 0, :] = ac[:, 1, :]
                # tmp1[:, 1, :] = ac[:, 0, :]
                # np.save(filenames[idxp], tmp1)
            timeLag = np.arange(y.size)
            popt, pcov = curve_fit(func, timeLag, y)
            estTau_dec[idxp] = popt[1]
            estTau_error[idxp] = np.sqrt(np.diag(pcov))[1]
            tt = np.linspace(0, 1000, 2000)
            fittedFunc = func(tt, *popt)
            plt.plot(timeLag, y, 'ko-')
            plt.plot(ac[:50, neuronType], 'go-')
            plt.plot(tt, fittedFunc, 'r')
            plt.title('p = %s, tau = %s'%(bidirp*.1, kTau))
            plt.title('tau = %s, tau_dec = %.5s, p = %s, validIdx= %s'%(kTau, popt[1], bidirp*.1, validStartIdx))
            plt.xlim([-10, 50])
            # plt.waitforbuttonpress()
            # plt.clf()
        plt.savefig('./figs/tau%s_p%s_%s'%(kTau, p, neuronTag))
        estTau_dec_list.append(estTau_dec)
        estTauError_dec_list.append(estTau_error)           
        plt.clf()
    plt.figure()
    for kk, kTau in enumerate(tau_s):    
#        plt.plot(realp, estTau_dec[kk, :], 'r.-', label = r'$\tau_s = %sms$'%(kTau), linewidth = 0.2)
        plt.plot(p_list[kk], estTau_dec_list[kk], 'r.-', label = r'$\tau_s = %sms$'%(kTau), linewidth = 0.2)        
    plt.legend(loc = 0, numpoints = 1, frameon = False, prop = {'size':8})
    plt.ylabel(r'$\tau_{dec}(ms)$')
    plt.savefig('./figs/summary0')
    axPos = [0.25, 0.3, .65, .65]
    plt.yticks([0, 150, 300])
    plt.xticks([0, 0.5, 1.0])
    plt.xlabel('p')
    pdf.Print2Pdf(plt.gcf(), figFolder + 'summary_0_%s'%(neuronTag), paperSize, figFormat = 'eps', IF_ADJUST_POSITION = True,  axPosition = axPos, labelFontsize = 10, tickFontsize = 8)
    plt.figure()
    for kk, kTau in enumerate(tau_s):        
#        plt.plot(realp, estTau_dec[kk, :] / float(kTau), 'k.-', label = r'$\tau_s = %sms$'%(kTau), linewidth = 0.2)
        plt.plot(p_list[kk], estTau_dec_list[kk] / float(kTau), 'k.-', label = r'$\tau_s = %sms$'%(kTau), linewidth = 0.2)
    # plt.plot(realp, estTau_dec[1, :] / 48.0, '-o', label = r'$\tau_s = 48ms$')
 #   plt.legend(loc = 0)    
    plt.ylabel(r'$\tau_{dec} / \tau_s$')
    plt.savefig('./figs/summary1')
    axPos = [0.2, 0.3, .65, .65]
    plt.yticks([0, 40, 80])
    plt.xticks([0, 0.5, 1.0])
    plt.xlabel('p')    
    pdf.Print2Pdf(plt.gcf(), figFolder + 'summary_1_%s'%(neuronTag), paperSize, figFormat = 'eps', IF_ADJUST_POSITION = True,  axPosition = axPos, labelFontsize = 10, tickFontsize = 8)
    plt.close('all')
    plt.figure()
    axPos = [0.28, 0.31, .64, 0.64] #[0.25, 0.3, .65, .65]    
    for kk, kTau in enumerate(tau_s):        
#        plt.loglog(1.0 - np.array(realp), estTau_dec[kk, :] / float(kTau),'k.-', linewidth = 0.2)
        print kk, kTau
        plt.loglog(1.0 - np.array(p_list[kk]), estTau_dec_list[kk] / float(kTau), '.-', linewidth = 0.2, markersize = 0.5)
    if neuronTag == 'I':
        plt.xlabel('1-p')
    else:
        axPos = [0.28, 0.28, .64, 0.64]
    plt.ylabel(r'$\tau_{dec} / \tau_s$')
    pdf.Print2Pdf(plt.gcf(), figFolder + 'summary_loglog_normalized_by_tau_s_%s'%(neuronTag), paperSize, figFormat = 'eps', IF_ADJUST_POSITION = True,  axPosition = axPos, labelFontsize = 10, tickFontsize = 8)
    plt.figure()
    axPos = [0.28, 0.31, .64, 0.64]    # restore balance
    for kk, kTau in enumerate(tau_s):        
        # plt.loglog(1.0 - np.array(realp), estTau_dec[kk, :], 'r.-')
        plt.loglog(1.0 - np.array(p_list[kk]), estTau_dec_list[kk], 'r.-')
    if neuronTag == 'I':
        plt.xlabel('1-p')
    plt.ylabel(r'$\tau_{dec}(ms)$')    
    pdf.Print2Pdf(plt.gcf(), figFolder + 'summary_loglog_%s'%(neuronTag), paperSize, figFormat = 'eps', IF_ADJUST_POSITION = True,  axPosition = axPos, labelFontsize = 10, tickFontsize = 8)
#    np.save('est_tau_dec_%s'%(taudecayFiletag), [np.array(realp), estTau_dec])
np.save('est_tau_dec_%s'%(neuronTag), [np.array(p_list), estTau_dec_list, estTauError_dec_list])
plt.close('all')

# PLOT EXP FIT ONLY FOR TAU = 3ms
td = np.load('est_tau_dec_E.npy')
plt.errorbar(td[0][0][:], td[1][0][:]/3.0, yerr = td[2][0][:], fmt = '.k-', ecolor = 'k', elinewidth = 0.2, linewidth = 0.2, markersize = 2.0, markeredgewidth = 0.1)
td = np.load('est_tau_dec_I.npy')
plt.errorbar(td[0][0][:], td[1][0][:]/3.0, yerr = td[2][0][:], fmt = '.r-', ecolor = 'r', elinewidth = 0.2, linewidth = 0.2, markersize = 2.0 , markeredgewidth = 0.1)
plt.legend({'E', 'I'}, frameon = False, loc = 2, numpoints = 1, prop = {'size': 8})
plt.xlabel('p')
plt.ylabel(r'$\tau_{dec} / \tau_s$')
plt.yticks([0, 45, 90])
plt.xticks([0, 0.5, 1.0])
filename = 'est_tau_dec_ov_taus_0'
figFormat = 'eps'
axPosition = [0.26, 0.28, .65, 0.65]
pdf.Print2Pdf(plt.gcf(),  figFolder + filename,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=10, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)
plt.close('all');

# loglog for tau=3ms
# td = np.load('est_tau_dec_E.npy')
# plt.loglog(1 - np.array(td[0][0][:]), td[1][0][:] / 3.0, '.k-', linewidth = 0.2, markersize = 2.0, markeredgewidth = 0.1)
# td = np.load('est_tau_dec_I.npy')
# plt.loglog(1 - np.array(td[0][0][:]), td[1][0][:] / 3.0, '.r-', linewidth = 0.2, markersize = 2.0 , markeredgewidth = 0.1)
# plt.xlabel('1-p')
# plt.ylabel(r'$\tau_{dec} / \tau_s$')
# #plt.yticks([0, 45, 90])
# plt.xticks([0, 0.5, 1.0])
# filename = 'est_tau_dec_ov_taus_0_loglog'
# figFormat = 'eps'
# axPosition = [0.26, 0.28, .65, 0.65]
# pdf.Print2Pdf(plt.gcf(),  figFolder + filename,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=10, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)


plt.close('all');
fgE, axE = plt.subplots()
fgI, axI = plt.subplots()
tdE = np.load('est_tau_dec_E.npy')
tdI = np.load('est_tau_dec_E.npy')
pcolor = ('b', 'g', 'r', 'k', 'c')
#for kk, kTau in enumerate(tau_s):
#    axE.errorbar(tdE[0][kk][:-1], tdE[1][kk][:-1]/kTau, yerr = tdE[2][kk][:-1] / kTau, color = pcolor[kk], fmt = '.-', ecolor = pcolor[kk], elinewidth = 0.2, linewidth = 0.2, markersize = 2.0, markeredgewidth = 0.1)
 #   axI.errorbar(tdI[0][kk][:-1], tdI[1][kk][:-1]/kTau, yerr = tdI[2][kk][:-1] / kTau, color = pcolor[kk], fmt = '.-', ecolor = pcolor[kk], elinewidth = 0.2, linewidth = 0.2, markersize = 2.0, markeredgewidth = 0.1)
 #   plt.clf()
axE.set_xlabel('p')
axE.set_ylabel(r'$\tau_{dec} / \tau_s$')
axI.set_xlabel('p')
axI.set_ylabel(r'$\tau_{dec} / \tau_s$')
axE.set_yticks([0, 45, 90])
axI.set_yticks([0, 45, 90])
axE.set_xticks([0, 0.5, 1.0])
axI.set_xticks([0, 0.5, 1.0])
filenameE = 'tdec_vs_tsyn_E'
filenameI = 'tdec_vs_tsyn_I'
figFormat = 'eps'
axPosition = [0.26, 0.28, .65, 0.65]
pdf.Print2Pdf(fgE,  figFolder + filenameE,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=10, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)
pdf.Print2Pdf(fgI,  figFolder + filenameI,  paperSize, figFormat=figFormat, labelFontsize = 10, tickFontsize=10, titleSize = 10.0, IF_ADJUST_POSITION = True, axPosition = axPosition)    

  
