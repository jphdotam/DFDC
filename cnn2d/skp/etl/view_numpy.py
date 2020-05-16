import numpy as np
import matplotlib.pyplot as plt
import glob


def view(x, nrows=4, ncols=4):
    indices = np.linspace(0, len(x)-1, 16) 
    for ind, i in enumerate(indices):
        plt.subplot(4,4,ind+1)
        plt.imshow(x[int(i)])
    plt.show()


numpys = glob.glob('/users/ipan/downloads/numpies/*.npy')
numpys = [np.load(_) for _ in numpys]

for i in range(len(numpys)):
    view(numpys[i])



