import numpy as np
import pylab as plt


def SetProperties(ax, xLim, yLim, xLabel, yLabel):
    ax.set_xlim(*xLim)
    ax.set_xticks([xLim[0], np.sum(xLim) * 0.5, xLim[1]])
    ax.set_ylim(*yLim)
    ax.set_yticks([yLim[0], np.sum(yLim) * 0.5, yLim[1]])
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    
