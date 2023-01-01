"""
PRIA version
Additive white Gaussian noise (AWGN)
"""

from keras.datasets import mnist
from keras.utils import to_categorical
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np


# generate noise
def wgn(x, snr):
    """
    SNR: signal to noise ratio
        https://en.wikipedia.org/wiki/Signal-to-noise_ratio
    """
    _shape = x.shape
    x = x.ravel()
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    return (np.random.randn(len(x)) * np.sqrt(npower)).reshape(_shape)


# add noise to x
def awgn(x, snr=5, clip=[0, 1]):
    """
    x: value in [0,1] instead of [0,255]
    """
    min_, max_ = clip
    return np.clip(x + wgn(x, snr), min_, max_)


# convert raw dataset to noise dataset
def to_noise_dataset(X, snr, clip=[0, 1]):
    """
    X: value in [0,1] instead of [0,255]
    """
    return np.array([awgn(x, snr, clip) for x in X], dtype=X.dtype)


# convert mnist to awgn mnise
def build_mnist_awgn(snr):
    """
    return value in [0,255]
    """
    (raw_x_train, raw_y_train), (raw_x_test, raw_y_test) = mnist.load_data()
    raw_x_train = raw_x_train.astype('float')
    raw_x_test = raw_x_test.astype('float')
    raw_x_train /= 255
    raw_x_test /= 255

    x_train = (to_noise_dataset(raw_x_train, snr) * 255).astype(np.uint8)
    y_train = raw_y_train
    x_test = (to_noise_dataset(raw_x_test, snr) * 255).astype(np.uint8)
    y_test = raw_y_test
    return (x_train, y_train), (x_test, y_test)


# %% load local datset function
def load_mnist_noise(path):
    if path[-4:] == '.mat':  # matlab file [1]
        f = sio.loadmat(path)
        x_train, y_train = f['train_x'], f['train_y']
        x_test, y_test = f['test_x'], f['test_y']
    elif path[-4:] == '.npz':  # numpy zip
        f = np.load(path)
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    else:
        raise ValueError('only support `.mat` or `.npz` file')
    return (x_train, y_train), (x_test, y_test)


# show example images
def showimg(img, cmap='gray'):
    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.show()


def showimgs(imgs, nrows, ncols, figsize=(12, 12), cmap='gray'):
    print('imgs shape:', imgs.shape)
    assert (imgs.shape[0] == nrows * ncols)
    newshape = (nrows, ncols) + imgs.shape[1:]
    imgs = imgs.reshape(newshape)

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    # fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    for i in range(nrows):
        for j in range(ncols):
            img = imgs[i, j, ...]
            axs[i, j].axis('off')
            axs[i, j].imshow(img, cmap=cmap)
    plt.show()


# %% run awgn
if __name__ == '__main__':
    # SNR = 3  #signal noise ratio, smaller value then stronger noise
    SNR = 5
    # SNR = 0.5

    # load images as (?, 784) shape
    # fname_mat = 'mnist-pria-awgn.mat'
    fname_npz = 'mnist-pria-awgn_snr=%s.npz' % str(SNR)
    save_compressed = False

    if os.path.exists(fname_npz):
        (x_train, y_train), (x_test, y_test) = load_mnist_noise(fname_npz)
    else:
        print('building noise mnist...')
        (x_train, y_train), (x_test, y_test) = build_mnist_awgn(snr=SNR)
        print('save noise mnist...')
        if save_compressed:
            np.savez_compressed(fname_npz, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        else:
            np.savez(fname_npz, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

# show first 16 images
if __name__ == '__main__':
    show_size = (8, 8)
    m = 4
    m2 = m * m

    (raw_x_train, raw_y_train), (raw_x_test, raw_y_test) = mnist.load_data()
    raw_x_train_show = raw_x_train[:m2].reshape(-1, 28, 28)
    showimgs(raw_x_train_show, m, m, figsize=show_size)
    x_train_show = x_train[:m2].reshape(-1, 28, 28)
    showimgs(x_train_show, m, m, figsize=show_size)

    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
