import pickle
from matplotlib import pyplot as plt

with open('./pong_autoencoder_loss.txt', 'rb') as fp:
    loss = pickle.load(fp)
    plt.plot(loss)
    plt.ylabel('AE Loss')
    plt.xlabel('Epochs (100s)')
    plt.show()