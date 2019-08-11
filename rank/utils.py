# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

import numpy as np
import six
import pickle
import scipy
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import serializers
from tqdm import tqdm
import scipy.stats as ss
from sklearn.preprocessing import StandardScaler



class NN(object):

    def loadModel(self, modelName):
        print('Load model')
        serializers.load_hdf5(modelName, self.model)
        print('Load optimizer state')
        serializers.load_hdf5(modelName[:-5] + 'state', self.optimizer)


    def initializeModel(self, Model, train_X, n_units1, n_units2, optimizerAlgorithm):
        print("prepare initialized model!")
        self.model = Model(len(train_X[0]), n_units1, n_units2, 1)
        self.initializeOptimizer(optimizerAlgorithm)

    def initializeOptimizer(self, optimizerAlgorithm):
        if optimizerAlgorithm == "Adam":
            self.optimizer = optimizers.Adam()
        elif optimizerAlgorithm == "AdaGrad":
            self.optimizer = optimizers.AdaGrad()
        elif optimizerAlgorithm == "SGD":
            self.optimizer = optimizers.MomentumSGD()
        else:
            raise ValueError('could not find %s in optimizers {"Adam", "AdaGrad", "SGD"}' % (optimizerAlgorithm))
        self.optimizer.setup(self.model)

    def saveModels(self, savemodelName):
        print('save the model')
        serializers.save_hdf5(savemodelName, self.model)
        print('save the optimizer')
        serializers.save_hdf5(savemodelName[:-5]+ 'state', self.optimizer)

    def splitData(self, fit_X, fit_y, tv_ratio):
        print('load dataset')
        perm = np.random.permutation(len(fit_X))
        N_train = int(np.floor(len(fit_X) * tv_ratio))
        train_X, validate_X = np.split(fit_X[perm].astype(np.float32),   [N_train])
        train_y, validate_y = np.split(fit_y[perm].astype(np.float32).reshape(len(fit_y), 1), [N_train])
        return train_X, train_y, validate_X, validate_y

    def predictTargets(self, x_pred, batchsize):
        N_pred = len(x_pred)
        y_pred = np.zeros(0)
        for j in tqdm(six.moves.range(0, N_pred, batchsize)):
            x = chainer.Variable(np.asarray(x_pred[j:j + batchsize]), volatile='on')
            y_pred = np.append(y_pred, self.model.predict(x))
        return y_pred

    def predict(self, predict_X):
        return self.model.predict(predict_X.astype(np.float32))

    # def predict(self, predict_X, batchsize=100):
    #     return self.predictTargets(predict_X.astype(np.float32), batchsize)


# -*- coding: utf-8 -*-
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


######################################################################################
# 誤差のプロット関数。自動的に保存するので上書きされたくない時は名前を変える

def acc(train_acc, test_acc, savename='result_acc.pdf'):
    ep = np.arange(len(train_acc)) + 1

    plt.plot(ep, train_acc, color="blue", linewidth=1, linestyle="-", label="Train")
    plt.plot(ep, test_acc, color="red", linewidth=1, linestyle="-", label="Test")
    plt.title("Accuracy")
    plt.xlabel("iteration")
    plt.ylabel("accuracy")
    plt.legend(loc='lower right')
    plt.savefig(savename)


def loss(train_loss, test_loss, savename='result_loss.pdf'):
    ep = np.arange(len(train_loss)) + 1

    plt.plot(ep, train_loss, color="blue", linewidth=1, linestyle="-", label="Train")
    plt.plot(ep, test_loss, color="red", linewidth=1, linestyle="-", label="Test")
    plt.title("Loss")
    plt.xlabel("iteration")
    plt.ylabel("loss")

    plt.legend(loc='upper right')
    plt.savefig(savename)