import theano
import theano.tensor as tensor
import numpy as np
import cPickle as pickle 
import timeit
import random
import time
import lasagne
import sys

import subprocess
from subprocess import Popen, PIPE, STDOUT

random.seed(1)
np.random.seed(1)
eps = 1e-8


def init_xavier_uniform(n_in, n_out):
	import math
	a = math.sqrt(2.0/(n_in + n_out))
	W = np.random.uniform( -a, a,  (n_in, n_out)).astype(theano.config.floatX)
	return W



def get_minibatches_idx(n, minibatch_size, shuffle=False):
        idx_list = np.arange(n, dtype="int32")

        if shuffle:
            np.random.shuffle(idx_list)

        minibatches = []
        minibatch_start = 0
        for i in range(n // minibatch_size):
            minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
            minibatch_start += minibatch_size

        if (minibatch_start != n):
            # Make a minibatch out of what is left
            minibatches.append(idx_list[minibatch_start:])

        return zip(range(len(minibatches)), minibatches)



class LSTM(object):
	def __init__(self, hidden_size, backwards=False):

		self.hidden_size = hidden_size
		self.backwards = backwards

		# lstm W matrixes, Wf, Wi, Wo, Wc respectively, all config.floatX type
		self.W = theano.shared(name="W", value=init_xavier_uniform(self.hidden_size, 4*self.hidden_size), borrow=True)
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
                        h0 = tensor.alloc(np.asarray(0., dtype=theano.config.floatX), batch_size, self.hidden_size)
                if C0 == None:
                        C0 = tensor.alloc(np.asarray(0., dtype=theano.config.floatX), batch_size, self.hidden_size)

                def _step( m, X, h_, C_, W, U, b, w_ci, w_cf, w_co):
                        bfr_actv = tensor.dot(X, W) + b+  tensor.dot(h_, U)

                        f = tensor.nnet.sigmoid( bfr_actv[:, 0:self.hidden_size] + C_*(w_cf.dimshuffle('x',0)) )                     #forget gate (batch_size, hidden_size)
                        i = tensor.nnet.sigmoid( bfr_actv[:, 1*self.hidden_size:2*self.hidden_size] + C_*(w_ci.dimshuffle('x',0)))   #input gate (batch_size, hidden_size)
                        Cp = tensor.tanh( bfr_actv[:, 3*self.hidden_size:4*self.hidden_size] )        #candi states (batch_size, hidden_size)



                        C = i*Cp + f*C_
                        C = m[:, None]*C + (1.0 - m)[:, None]*C_

                        o = tensor.nnet.sigmoid( bfr_actv[:, 2*self.hidden_size:3*self.hidden_size]+ Cp*(w_co.dimshuffle('x',0)) ) #output  gate (batch_size, hidden_size)

                        h = o*tensor.tanh( C )
                        h = m[:, None]*h + (1.0 - m)[:, None]*h_

                        h, C = tensor.cast(h, theano.config.floatX), tensor.cast(h, theano.config.floatX)

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







class Seq2Seq(object):
	def __init__(self, We, params):

		lstm_layers_num = 1
		en_hidden_size = We.shape[1]
		self.eta = params.eta
		self.num_labels = params.num_labels
		self.en_hidden_size = en_hidden_size
		self.de_hidden_size = params.de_hidden_size

		self.lstm_layers_num = params.lstm_layers_num
		self._train = None
		self._utter = None
		self.params = []
		self.encoder_lstm_layers = []
		self.decoder_lstm_layers = []
		self.hos = []
		self.Cos = []

		encoderInputs = tensor.imatrix()
		decoderInputs, decoderTarget = tensor.imatrices(2)
		encoderMask, TF, decoderMask, decoderInputs0, diagonal = tensor.fmatrices(5)


		self.lookuptable = theano.shared(We)

		#### the last one is for the stary symbole
		self.de_lookuptable = theano.shared(name="Decoder LookUpTable", value=init_xavier_uniform(self.num_labels +1, self.de_hidden_size), borrow=True)
			
		self.linear = theano.shared(name="Linear", value = init_xavier_uniform(self.de_hidden_size + 2*en_hidden_size, self.num_labels), borrow= True)
		self.linear_bias = theano.shared(name="Hidden to Bias", value=np.asarray(np.random.randn(self.num_labels, )*0., dtype=theano.config.floatX), borrow=True)                     
  
		self.hidden_decode = theano.shared(name="Hidden to Decode", value= init_xavier_uniform(2*en_hidden_size, self.de_hidden_size), borrow = True)

		self.hidden_bias = theano.shared(
                        name="Hidden to Bias",
                        value=np.asarray(np.random.randn(self.de_hidden_size, )*0., dtype=theano.config.floatX) ,
                        borrow=True
                        )		

	
		#self.params += [self.linear, self.de_lookuptable, self.hidden_decode, self.hidden_bias]    #concatenate
		self.params += [self.linear, self.linear_bias , self.de_lookuptable, self.hidden_decode, self.hidden_bias]    #the initial hidden state of decoder lstm is zeros
		#(max_sent_size, batch_size, hidden_size)
		state_below = self.lookuptable[encoderInputs.flatten()].reshape((encoderInputs.shape[0], encoderInputs.shape[1], self.en_hidden_size))
		for _ in range(self.lstm_layers_num):
			
			enclstm_f = LSTM(self.en_hidden_size)
			enclstm_b = LSTM(self.en_hidden_size, True)
			self.encoder_lstm_layers.append(enclstm_f)    #append
			self.encoder_lstm_layers.append(enclstm_b)    #append
			self.params += enclstm_f.params + enclstm_b.params   #concatenate
			
			hs_f, Cs_f = enclstm_f.forward(state_below, encoderMask)
			hs_b, Cs_b = enclstm_b.forward(state_below, encoderMask)
			
			hs = tensor.concatenate([hs_f, hs_b], axis=2)
			Cs = tensor.concatenate([Cs_f, Cs_b], axis=2)
			hs0 = tensor.concatenate([hs_f[-1], hs_b[0]], axis=1)
			Cs0 = tensor.concatenate([Cs_f[-1], Cs_b[0]], axis=1) 			
			#self.hos += tensor.tanh(tensor.dot(hs0, self.hidden_decode) + self.hidden_bias),
			#self.Cos += tensor.tanh(tensor.dot(Cs0, self.hidden_decode) + self.hidden_bias),
			self.hos += tensor.alloc(np.asarray(0., dtype=theano.config.floatX), encoderInputs.shape[1], self.de_hidden_size),			
			self.Cos += tensor.alloc(np.asarray(0., dtype=theano.config.floatX), encoderInputs.shape[1], self.de_hidden_size),
			state_below = hs

		Encoder = state_below
		Encoder_shuffle = Encoder.dimshuffle(1, 0, 2)
		## Encoder_shuffle : B x L x h_e
		##Encoder_shuffle_re : B x L x h_d
		Encoder_shuffle_re = tensor.dot(Encoder_shuffle, self.hidden_decode)+ self.hidden_bias[None, None, :]

		state_below = self.de_lookuptable[decoderInputs.flatten()].reshape((decoderInputs.shape[0], decoderInputs.shape[1], self.de_hidden_size))
		for i in range(self.lstm_layers_num):
			declstm = LSTM(self.de_hidden_size)
			self.decoder_lstm_layers += declstm,    #append
			self.params += declstm.params    #concatenate
			ho, Co = self.hos[i], self.Cos[i]
			state_below, Cs = declstm.forward(state_below, decoderMask, ho, Co)
	
		##### Here we include the representation from the decoder	
		decoder_lstm_outputs = tensor.concatenate([state_below, Encoder], axis=2)

		ei, di, dt = tensor.imatrices(3)    #place holders
		em, dm, tf, di0, diag =tensor.fmatrices(5) 
		#####################################################
		#####################################################
		linear_outputs = tensor.dot(decoder_lstm_outputs, self.linear) + self.linear_bias[None, None, :]
		softmax_outputs, updates = theano.scan(
			fn=lambda x: tensor.nnet.softmax(x),
			sequences=[linear_outputs],
			)

		def _NLL(pred, y, m):
			return -m * tensor.log(pred[tensor.arange(encoderInputs.shape[1]), y])

		costs, _ = theano.scan(fn=_NLL, sequences=[softmax_outputs, decoderTarget, decoderMask])
		loss = costs.sum() / decoderMask.sum() + params.L2*sum(lasagne.regularization.l2(x) for x in self.params)

		updates = lasagne.updates.adam(loss, self.params, self.eta)
        	#updates = lasagne.updates.apply_momentum(updates, self.params, momentum=0.9)

		###################################################
		#### using the ground truth when training
		##################################################
		self._train = theano.function(
			inputs=[ei, em, di, dm, dt],
			outputs=[loss, softmax_outputs],
			updates=updates,
			givens={encoderInputs:ei, encoderMask:em, decoderInputs:di, decoderMask:dm, decoderTarget:dt}
			)


		#########################################################################
		### For schedule sampling
		#########################################################################
	
		
		###### always use privous predict as next input 
                def _step2(diag_, state_, hs_, Cs_):

                        hs, Cs = [], []
                        token_idxs = tensor.cast(state_.argmax(axis=-1), "int32" )
			msk_ = tensor.fill( (tensor.zeros_like(token_idxs, dtype="float32")), 1)
			msk_ = msk_.dimshuffle('x', 0)
                        state_below0 = self.de_lookuptable[token_idxs].reshape((1, encoderInputs.shape[1], self.de_hidden_size))
                        for i, lstm in enumerate(self.decoder_lstm_layers):
                                h, C = lstm.forward(state_below0, msk_, hs_[i], Cs_[i])    #mind msk
                                hs += h[-1],
                                Cs += C[-1],
                                state_below0 = h

                        hs, Cs = tensor.as_tensor_variable(hs), tensor.as_tensor_variable(Cs)
			state_below0 = state_below0.reshape((encoderInputs.shape[1], self.de_hidden_size))
	
			attn_weight = tensor.batched_dot(Encoder_shuffle_re, state_below0)			
			attn_weight = tensor.nnet.softmax(attn_weight)
			attn_weight *= (encoderMask.dimshuffle(1,0))			
		
			attn_weight *= (diag_.dimshuffle('x', 0))	
			####  normalize
			attn_weight = attn_weight/(tensor.sum(attn_weight, axis=1).dimshuffle(0,'x') + 1e-20)
                        ####### ctx_ : (b, h)                   
                        ctx_ = tensor.sum(Encoder_shuffle*attn_weight[:,:, None], axis=1)
			

			state_below0 = tensor.concatenate([ctx_, state_below0], axis =1)
                        newpred = tensor.dot(state_below0, self.linear) + self.linear_bias[None, :]
                        state_below = tensor.nnet.softmax(newpred)
			##### the beging symbole probablity is 0
                        extra_p = tensor.zeros_like(hs[:,:,0])
                        state_below = tensor.concatenate([state_below, extra_p.T], axis=1)
 
                        return state_below, hs, Cs
 

		hs0, Cs0 = tensor.as_tensor_variable(self.hos, name="hs0"), tensor.as_tensor_variable(self.Cos, name="Cs0")
		train_outputs, _ = theano.scan(
                        fn=_step2,
			sequences= [diagonal],
                        outputs_info=[decoderInputs0, hs0, Cs0],
                        n_steps=encoderInputs.shape[0]
                        )
		
		train_predict = train_outputs[0]
		train_costs, _ = theano.scan(fn=_NLL, sequences=[train_predict, decoderTarget, decoderMask])
		
                train_loss = train_costs.sum() / decoderMask.sum() + params.L2*sum(lasagne.regularization.l2(x) for x in self.params)
		#from adam import adam		
                #train_updates = adam(train_loss, self.params, self.eta)
                #train_updates = lasagne.updates.apply_momentum(train_updates, self.params, momentum=0.9)
		from momentum import momentum
                train_updates = momentum(train_loss, self.params, params.eta, momentum=0.9)
		
		self._train2 = theano.function(
                        inputs=[ei, em, di0, dm, dt, diag],
                        outputs=[train_loss, train_predict],
                        updates=train_updates,
                        givens={encoderInputs:ei, encoderMask:em, decoderInputs0:di0, decoderMask:dm, decoderTarget:dt, diagonal: diag}
			#givens={encoderInputs:ei, encoderMask:em, decoderInputs:di, decoderMask:dm, decoderTarget:dt, TF:tf}
                        )
		
		listof_token_idx = train_predict.argmax(axis=-1)
		self._utter = theano.function(
                        inputs=[ei, em, di0, diag],
                        outputs=listof_token_idx,
                        givens={encoderInputs:ei, encoderMask:em, decoderInputs0:di0, diagonal:diag}
                        )
		


	def prepare_decoder_data(self, seqs):

                lengths = [len(s) for s in seqs]
                n_samples = len(seqs)
                maxlen = np.max(lengths)
                sumlen = sum(lengths)

                x = np.zeros((maxlen, n_samples)).astype('int32')
                x_out = np.zeros((maxlen, n_samples)).astype('int32')
                x_mask = np.zeros((maxlen, n_samples)).astype(theano.config.floatX)


                for idx, s in enumerate(seqs):
			### -1 is start symbole of a sentence
			"""
                        x[:lengths[idx], idx] = [-1] + s[:-1][::-1]
                        x_mask[:lengths[idx], idx] = 1.
                        x_out[:lengths[idx], idx] = s[::-1]
			"""
			
			x[:lengths[idx], idx] = [-1] + s[:-1]
                        x_mask[:lengths[idx], idx] = 1.
                        x_out[:lengths[idx], idx] = s
			


                return x, x_mask, x_out


	def prepare_encoder_data(self, seqs, window):

                lengths = [len(s) for s in seqs]
                n_samples = len(seqs)
                maxlen = np.max(lengths)


                x = np.zeros((maxlen, n_samples)).astype('int32')
                x_mask = np.zeros((maxlen, n_samples)).astype(theano.config.floatX)
		diag = np.zeros((maxlen, maxlen)).astype(theano.config.floatX)

                for idx, s in enumerate(seqs):
                        x[:lengths[idx], idx] = s
                        x_mask[:lengths[idx], idx] = 1.

		for i in range(maxlen):
			for j in range(maxlen):
				if (abs(j-i)<=window):
					if (int(sys.argv[6])==0):
                                                diag[i,j]= window+1-abs(j-i)

                                        elif(int(sys.argv[6])==1):
                                                if (i==j):
                                                        diag[i,j] = window+1
                                                else:
                                                        diag[i,j]=1.0
                                        else:
                                                diag[i,j]= np.exp(-2*(j-i)**2/(window**2))
					

                return x, x_mask, diag


	def train(self, trainX, devX, testX, params):
		
		self.textfile = open(params.outfile, 'w')

		trainX0, trainX1 = trainX
                devX0, devX1 = devX
                testX0, testX1 = testX

                devx0, devx0mask, devdiag = self.prepare_encoder_data(devX0, params.window)
                devx1, _, devy1 = self.prepare_decoder_data(devX1)

                testx0, testx0mask, testdiag = self.prepare_encoder_data(testX0, params.window)
                testx1, _ , testy1 = self.prepare_decoder_data(testX1)
		
		
		
		bestdev = -1
                best_t =0
             
                try:
                        for eidx in xrange(30):
                                n_samples = 0


                                kf = get_minibatches_idx(len(trainX0), params.batchsize, shuffle=True)
                                uidx = 0
                                aa = 0
                                bb = 0
				start_time1 = time.time()
                                for _, train_index in kf:

                                        uidx += 1

                                        X0 = [trainX0[ii] for ii in train_index]
                                        X1 = [trainX1[ii] for ii in train_index]

                                        n_samples += len(train_index)

                                        x0, x0mask, diag = self.prepare_encoder_data(X0, params.window)
                                        x1_in, _ , x1_out = self.prepare_decoder_data(X1)
					
		
					
					#if (uidx%2==1):
                                        #	traincost, _ = self._train(x0, x0mask, x1_in, x0mask, x1_out)
					#else:
					x1_in2 = np.zeros((len(train_index), self.num_labels+1), dtype = 'float32')
					x1_in2[:,-1] = 1.
					traincost, _ = self._train2(x0, x0mask, x1_in2, x0mask, x1_out, diag)
				       	
					#traincost, _ = self._train(x0, x0mask, x1_in, x0mask, x1_out)
					
 
					#print 'traincost', traincost	
                                        if np.isnan(traincost):
                                                self.textfile.write("NaN detected \n")
                                                self.textfile.flush()
				end_time1 = time.time()
                                self.textfile.write("epoches %d trainig time %f \n" %(eidx+1, end_time1 - start_time1))
				self.textfile.flush()


                                devpred  = self.utter(devx0, devx0mask, devdiag)
				f = open('pred_ner_inf_' + params.outfile, 'w')
                                devlength = [len(s) for s in devX0]
                                aaaa = 0
                                for ii, devl in enumerate(devlength):
                                        for jj in range(devl):
                                                eva_string = params.devrawx[ii][jj] + ' '+ params.devpos[ii][jj] + ' '+ params.taggerlist[devy1[jj, ii]] +' ' +  params.taggerlist[devpred[jj, ii]]
                                                #eva_string =  'John NNP ' + params.taggerlist[devy0[aaaa]] +' ' +  params.taggerlist[dev_pred[aaaa]]
                                                if len(eva_string.split(' '))!=4:
                                                        print   len(eva_string.split(' '))  , eva_string
                                                f.write(eva_string + '\n'  )
                                                aaaa +=1
                                        f.write('\n')
                                f.close()
                                cmd = 'perl conlleval < pred_ner_inf_' + params.outfile
                                p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
                                output = p.stdout.read()
                                output =  output.split('\n')
                                output = output[1]
                                output = output.split(' ')
                                dev_f10 = float(output[-1])
		
				                                
                                if bestdev < dev_f10:
                                        bestdev = dev_f10
                                        best_t = eidx
                                        #tmp_a_para = [p.get_value() for p in self.params]
                                        #saveParams( tmp_a_para , params.outfile + '.pickle')
					testpred  = self.utter(testx0, testx0mask, testdiag)
					f = open('pred_ner_inf_' + params.outfile, 'w')
                                	testlength = [len(s) for s in testX0]
                                	for ii, testl in enumerate(testlength):
                                        	for jj in range(testl):
                                                	eva_string = params.testrawx[ii][jj] + ' '+ params.testpos[ii][jj] + ' '+ params.taggerlist[testy1[jj, ii]] +' ' +  params.taggerlist[testpred[jj, ii]]

                                                	f.write(eva_string + '\n'  )
                                        	f.write('\n')
                                	f.close()
                                	cmd = 'perl conlleval < pred_ner_inf_' + params.outfile
                                	p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
                                	output = p.stdout.read()
                                	output =  output.split('\n')
                                	output = output[1]
                                	output = output.split(' ')
                                	test_f10 = float(output[-1])
                                	#print 'epoch ', (eidx),  'test f1', test_f10
                                	self.textfile.write("epoch %d  devf1 %f  testf1 %f\n" %(eidx , dev_f10, test_f10))
                                	self.textfile.flush()


                                	print 'prediction', devpred[:, 0] , 'devf ', dev_f10, 'testf ', test_f10

                except KeyboardInterrupt:
                        #print "Classifer Training interupted"
                        self.textfile.write( 'classifer training interrupt \n')
                        self.textfile.flush()
		print 'bestdev', bestdev, 'best_t', best_t
                self.textfile.write("best dev acc: %f  at time %d \n" % (bestdev, best_t))
                self.textfile.close()


		###return self._train(encoderInputs, encoderMask, decoderInputs, decoderMask, decoderTarget)

	def utter(self, encoderInputs, encoderMask, diagonal):
		n_samples = encoderInputs.shape[1]
		decoderInputs=np.zeros((n_samples, self.num_labels+1),  dtype='float32')
		decoderInputs[:, -1] = 1.
		#if encoderInputs.ndim == 1:
		#	encoderInputs = encoderInputs.reshape((encoderInputs.shape[0], 1))
		#	encoderMask = encoderMask.reshape((encoderMask.shape[0], 1))
		rez = self._utter(encoderInputs, encoderMask ,decoderInputs, diagonal)
		#return rez.reshape((encoderMask.shape[0], encoderMask.shape[1]))
		return rez


	def acc(self, pred, rez, mask):
		pg = np.equal(pred, rez)
		acc = np.sum(pg)
		


