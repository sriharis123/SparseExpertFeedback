import pickle
from matplotlib import pyplot as plt

with open('./breakout_gamma.txt', 'rb') as fp:
    with open('./breakout_full_uniform.txt', 'rb') as fp2:
        f, axarr = plt.subplots(1,2)
        axarr[0].plot(pickle.load(fp))
        axarr[1].plot(pickle.load(fp2))
        # f.ylabel('Breakout Loss')
        # f.xlabel('Epochs (10s)')
        plt.show()