from . import lexicons
from . import util
import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, product
import tensorflow as tf
#from keras import backend as K
import tensorflow.keras.backend as K
#from keras.models import Graph
from keras.layers.core import Dense, Lambda
from tensorflow.keras.optimizers import Adam, Optimizer, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from keras.regularizers import Regularizer
from keras.constraints import Constraint
import keras
from keras.layers import Input, Dense, Lambda, Activation

import theano.tensor as T
from .representations.embedding import Embedding


"""
Helper methods for learning transformations of word embeddings.
"""

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Optimizer

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Optimizer

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Optimizer

    
class Orthogonal(Constraint):
    def __call__(self, p):
        print("here")
        s, u, v  = tf.linalg.svd(p)
        return K.dot(u,K.transpose(v))

class OthogonalRegularizer(Regularizer):
    def __init__(self, strength=0.):
        self.strength = strength

    def set_param(self, p):
        self.p = p

    def __call__(self, loss):
        loss += K.sum(K.square(self.p.dot(self.p.T) - T.identity_like(self.p))) * self.strength
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__,
                "strength": self.strength}


def orthogonalize(Q):
    U, S, V = np.linalg.svd(Q)
    return U.dot(V.T)


class DatasetMinibatchIterator:
    def __init__(self, embeddings, positive_seeds, negative_seeds, batch_size=512, **kwargs):
        self.words, embeddings1, embeddings2, labels = [], [], [], []

        def add_examples(word_pairs, label):
            for w1, w2 in word_pairs:
                embeddings1.append(embeddings[w1])
                embeddings2.append(embeddings[w2])
                labels.append(label)
                self.words.append((w1, w2))

        add_examples(combinations(positive_seeds, 2), 1)
        add_examples(combinations(negative_seeds, 2), 1)
        add_examples(product(positive_seeds, negative_seeds), -1)
        self.e1 = np.vstack(embeddings1)
        self.e2 = np.vstack(embeddings2)
        self.y = np.array(labels)

        self.batch_size = batch_size
        self.n_batches = (self.y.size + self.batch_size - 1) // self.batch_size

    def shuffle(self):
        perm = np.random.permutation(np.arange(self.y.size))
        self.e1, self.e2, self.y, self.words = \
            self.e1[perm], self.e2[perm], self.y[perm], [self.words[i] for i in perm]

    def __iter__(self):
        for i in range(self.n_batches):
            batch = np.arange(i * self.batch_size, min(self.y.size, (i + 1) * self.batch_size))
            yield {
                'embeddings1': self.e1[batch],
                'embeddings2': self.e2[batch],
                'y': self.y[batch][:, np.newaxis]
            }


def get_model(inputdim, outputdim, regularization_strength=0.01, lr=0.000, cosine=False, **kwargs):
    transformation = Dense(inputdim, kernel_initializer='identity', kernel_constraint=Orthogonal())

    embeddings1 = Input(shape=(inputdim,))
    embeddings2 = Input(shape=(inputdim,))
    transformed1 = transformation(embeddings1)
    transformed2 = transformation(embeddings2)

    projected1 = Lambda(lambda x: x[:, :outputdim])(transformed1)
    negprojected2 = Lambda(lambda x: -x[:, :outputdim])(transformed2)

    if cosine:
        normalized1 = Lambda(lambda x:  x / K.reshape(K.sqrt(K.sum(x * x, axis=1)), (x.shape[0], 1)))(projected1)
        negnormalized2 = Lambda(lambda x:  x / K.reshape(K.sqrt(K.sum(x * x, axis=1)), (x.shape[0], 1)))(negprojected2)
        distances =  Lambda(lambda x: K.reshape(K.sqrt(K.sum(x[0]*x[0], axis=1),(x[0].shape[0], 1))) * K.reshape(K.sqrt(K.sum(x[1]*x[1] , axis=1)), (x[1].shape[0], 1)))([normalized1, negnormalized2])
    else:
        
        distances= Lambda(lambda x: K.sqrt(K.sum(x[0]*x[0], axis=1, keepdims=True)) + K.sqrt(K.sum(x[1]*x[1] , axis=1, keepdims=True)))([projected1, negprojected2])

    model = keras.Model(inputs=[embeddings1, embeddings2], outputs=distances)

    lr_schudle=ExponentialDecay(
                    initial_learning_rate=5.0, decay_steps=1, decay_rate=0.99, staircase=False
    )
    model.compile(loss=lambda y, d: K.mean(tf.cast(y, tf.float64) * tf.cast(d, tf.float64)), 
                                optimizer=SGD(learning_rate=lr_schudle)
    )
    return model


def apply_embedding_transformation(embeddings, positive_seeds, negative_seeds,
                                   n_epochs=5, n_dim=10, force_orthogonal=False,
                                   plot=False, plot_points=50, plot_seeds=False,
                                   **kwargs):
    print("Preparing to learn embedding tranformation")
    dataset = DatasetMinibatchIterator(embeddings, positive_seeds, negative_seeds, **kwargs)
    model = get_model(embeddings.m.shape[1], n_dim, **kwargs)

    print("Learning embedding transformation")
#    prog = util.Progbar(n_epochs)
    for epoch in range(n_epochs):
        dataset.shuffle()
        loss = 0
        for i, X in enumerate(dataset):
            loss += model.train_on_batch([X["embeddings1"], X["embeddings2"]], y=X['y']) * X['y'].size
            Q, b = model.get_weights()
            if force_orthogonal:
                Q = orthogonalize(Q)
            model.set_weights([Q, np.zeros_like(b)])
#        prog.update(epoch + 1, exact_values=[('loss', loss / dataset.y.size)])
    Q, b = model.get_weights()
    new_mat = embeddings.m.dot(Q)[:,0:n_dim]
    #print "Orthogonality rmse", np.mean(np.sqrt(
    #    np.square(np.dot(Q, Q.T) - np.identity(Q.shape[0]))))

    if plot and n_dim == 2:
        plot_words = positive_seeds + negative_seeds if plot_seeds else \
            [w for w in embeddings if w not in positive_seeds and w not in negative_seeds]
        plot_words = set(random.sample(plot_words, plot_points))
        to_plot = {w: embeddings[w] for w in embeddings if w in plot_words}

        lexicon = lexicons.load_lexicon()
        plt.figure(figsize=(10, 10))
        for w, e in to_plot.iteritems():
            plt.text(e[0], e[1], w,
                     bbox=dict(facecolor='green' if lexicon[w] == 1 else 'red', alpha=0.1))
        xmin, ymin = np.min(np.vstack(to_plot.values()), axis=0)
        xmax, ymax = np.max(np.vstack(to_plot.values()), axis=0)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.show()
    return Embedding(new_mat, embeddings.iw, normalize=n_dim!=1)
