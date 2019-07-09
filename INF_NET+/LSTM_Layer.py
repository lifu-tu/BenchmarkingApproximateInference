import numpy as np
import theano
from theano import tensor as T
import lasagne
import random as random
import pickle
import cPickle
import time
import sys
from random import randint


random.seed(1)
np.random.seed(1)
eps = 0.0000001


def init_xavier_uniform(n_in, n_out):
        import math
        a = math.sqrt(2.0/(n_in + n_out))
        W = np.random.uniform( -a, a,  (n_in, n_out)).astype(theano.config.floatX)
        return W


class LSTM(object):
        def __init__(self, num_inputs, hidden_size, backwards=False):

                self.hidden_size = hidden_size
                self.backwards = backwards

                # lstm W matrixes, Wf, Wi, Wo, Wc respectively, all config.floatX type
                self.W = theano.shared(name="W", value=init_xavier_uniform(num_inputs, 4*self.hidden_size), borrow=True)
                # lstm U matrixes, Uf, Ui, Uo, Uc respectively, all config.floatX type
                self.U = theano.shared(name="U", value=init_xavier_uniform(self.hidden_size, 4*self.hidden_size), borrow=True)
                # lstm b vectors, bf, bi, bo, bc respectively, all config.floatX type
                self.b = theano.shared(name="b", value=np.zeros( 4*self.hidden_size, dtype=theano.config.floatX ), borrow=True)

                # peephole connection
                self.w_ci = theano.shared(name="c_i", value=np.zeros((self.hidden_size,), dtype=theano.config.floatX ), borrow=True)
                self.w_cf = theano.shared(name="c_f", value=np.zeros((self.hidden_size,), dtype=theano.config.floatX ), borrow=True)
                self.w_co = theano.shared(name="c_o", value=np.zeros((self.hidden_size,), dtype=theano.config.floatX ), borrow=True)

                self.params = [self.W, self.U, self.b, self.w_ci, self.w_cf, self.w_co]



        def forward(self, inputs, mask, h0=None, C0=None):
                """
                param inputs: #(max_sent_size, batch_size, hidden_size).
                inputs: state_below
                """
                if inputs.ndim == 3:
                        batch_size = inputs.shape[1]
                else:
                        batch_size = 1

                if h0 == None:
                        h0 = T.alloc(np.asarray(0., dtype=theano.config.floatX), batch_size, self.hidden_size)
                if C0 == None:
                        C0 = T.alloc(np.asarray(0., dtype=theano.config.floatX), batch_size, self.hidden_size)

                def _step( m, X,   h_, C_, W, U, b, w_ci, w_cf, w_co):
                        bfr_actv = T.dot(X, W) + b+  T.dot(h_, U)

                        f = T.nnet.sigmoid( bfr_actv[:, 0:self.hidden_size] + C_*(w_cf.dimshuffle('x',0)) )                     #forget gate (batch_size, hidden_size)
                        i = T.nnet.sigmoid( bfr_actv[:, 1*self.hidden_size:2*self.hidden_size] + C_*(w_ci.dimshuffle('x',0)))   #input gate (batch_size, hidden_size)
                        Cp = T.tanh( bfr_actv[:, 3*self.hidden_size:4*self.hidden_size] )        #candi states (batch_size, hidden_size)



                        C = i*Cp + f*C_
                        C = m[:, None]*C + (1.0 - m)[:, None]*C_

                        o = T.nnet.sigmoid( bfr_actv[:, 2*self.hidden_size:3*self.hidden_size]+ Cp*(w_co.dimshuffle('x',0)) ) #output  gate (batch_size, hidden_size)

                        h = o*T.tanh( C )
                        h = m[:, None]*h + (1.0 - m)[:, None]*h_
			
			h, C = T.cast(h, theano.config.floatX), T.cast(h, theano.config.floatX)

                        return h, C

                outputs, updates = theano.scan(
                        fn = _step,
                        sequences = [mask, inputs],
                        outputs_info = [h0, C0],
                        non_sequences = [self.W, self.U, self.b, self.w_ci, self.w_cf, self.w_co],
                        go_backwards = self.backwards
                        )

                hs, Cs = outputs
                if self.backwards:
                        hs = hs[::-1]
                        Cs = Cs[::-1]
                return hs, Cs

