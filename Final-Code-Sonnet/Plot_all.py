import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pylab import savefig

cost = [100, 500, 1000]
problem = ['quadratic','mnist','cifar','cifar-multi']
optimizer = ['DirectSave/SGD','DirectSave/RMSprop','DirectSave/Momentum','DirectSave/ADAM']
label = ['SGD','RMSprop','Momentum','Adam']

for i in cost:
    for j in problem:
        fig_path = os.getcwd() + '/FigOpt-' + str(i) + '/' + j
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        plotlosstrain = []
        plotlosseval = []
        count = 0
        for k in optimizer:
            slength = np.arange(i)
            plt.figure(1,figsize=(8, 5))
            txtpathtrain = os.getcwd() + '/Save-' + str(i) + '/' + j + '/' + k + '/plotlosstrain.out'
            with open(txtpathtrain) as f:
                for line in f:
                    val = line.split(",")
                    plotlosstrain.append([float(x) for x in val])
            plt.plot(slength, np.mean(plotlosstrain, 0), label=label[count])
            plt.xlabel('Steps')
            plt.ylabel('Training Loss (log10)')
            plt.legend()
            plt.figure(2, figsize=(8, 5))
            txtpatheval = os.getcwd() + '/Save-' + str(i) + '/' + j + '/' + k + '/plotlosseval.out'
            with open(txtpatheval) as f:
                for line in f:
                    val = line.split(",")
                    plotlosseval.append([float(x) for x in val])
            plt.plot(slength, np.mean(plotlosseval, 0), label=label[count])
            plt.xlabel('Steps')
            plt.ylabel('Validation Loss (log10)')
            plt.legend()
            count = count + 1
        plt.figure(1)
        savefig(fig_path + '/Training.png')
        plt.close()
        plt.figure(2)
        savefig(fig_path + '/Validation.png')
        plt.close()

