import theano
import theano.tensor as tensor
import numpy as np
import cPickle as pickle 
import timeit
import random

import lasagne

random.seed(1)
np.random.seed(1)

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
trng = RandomStreams(1234)

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





from LSTM_Layer import LSTM

Max_Char_Length = 30

def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=0.5, n=1,
                                     dtype=state_before.dtype),
        state_before * 0.5)
    return proj


class Seq2Seq(object):
	def __init__(self, We, char_embedd_table_initial, params):

		lstm_layers_num = 1
		emb_size = We.shape[1]
		self.eta = params.eta
		self.num_labels = params.num_labels
		self.en_hidden_size = params.en_hidden_size
		self.de_hidden_size = params.de_hidden_size

		self.lstm_layers_num = params.lstm_layers_num
		self._train = None
		self._utter = None
		self.params = []
		self.encoder_lstm_layers = []
		self.decoder_lstm_layers = []
		self.hos = []
		self.Cos = []

                char_embedd_dim = params.char_embedd_dim
                char_dic_size = len(params.char_dic)
                char_embedd_table = theano.shared(char_embedd_table_initial)
             

		encoderInputs = tensor.imatrix()
		decoderInputs, decoderTarget = tensor.imatrices(2)
		encoderMask, TF, decoderMask, decoderInputs0 = tensor.fmatrices(4)
                char_input_var = tensor.itensor3(name='char-inputs')
                ci = tensor.itensor3()

                use_dropout = tensor.fscalar()
                use_dropout0 = tensor.fscalar()

		self.lookuptable = theano.shared(We)

		#### the last index is for the stary symbole
		self.de_lookuptable = theano.shared(name="Decoder LookUpTable", value=init_xavier_uniform(self.num_labels +1, self.de_hidden_size), borrow=True)
			
		self.linear = theano.shared(name="Linear", value = init_xavier_uniform(self.de_hidden_size+2*self.en_hidden_size, self.num_labels), borrow= True)
                self.linear_bias = theano.shared(name="Hidden to Bias", value=np.asarray(np.random.randn(self.num_labels, )*0., dtype=theano.config.floatX), borrow=True)       
		##self.hidden_decode = theano.shared(name="Hidden to Decode", value= init_xavier_uniform(2*en_hidden_size, self.de_hidden_size), borrow = True)

		##self.hidden_bias = theano.shared(
                ##        name="Hidden to Bias",
                ##        value=np.asarray(np.random.randn(self.de_hidden_size, )*0., dtype=theano.config.floatX) ,
                ##        borrow=True
                ##        )		

	
		#self.params += [self.linear, self.de_lookuptable, self.hidden_decode, self.hidden_bias]    #concatenate
		self.params += [self.lookuptable, self.linear, self.linear_bias, self.de_lookuptable]    #the initial hidden state of decoder lstm is zeros
		#(max_sent_size, batch_size, hidden_size)
		state_below = self.lookuptable[encoderInputs.flatten()].reshape((encoderInputs.shape[0], encoderInputs.shape[1], emb_size))

                layer_char_input = lasagne.layers.InputLayer(shape=(None, None, Max_Char_Length ),
                                                     input_var=char_input_var, name='char-input')

                layer_char = lasagne.layers.reshape(layer_char_input, (-1, [2]))
                layer_char_embedding = lasagne.layers.EmbeddingLayer(layer_char, input_size=char_dic_size,
                                                             output_size=char_embedd_dim, W=char_embedd_table,
                                                             name='char_embedding')

                layer_char = lasagne.layers.DimshuffleLayer(layer_char_embedding, pattern=(0, 2, 1))


                # first get some necessary dimensions or parameters
                conv_window = 3
                num_filters = params.num_filters

                # construct convolution layer
                cnn_layer = lasagne.layers.Conv1DLayer(layer_char, num_filters=num_filters, filter_size=conv_window, pad='full',
                                           nonlinearity=lasagne.nonlinearities.tanh, name='cnn')
                # infer the pool size for pooling (pool size should go through all time step of cnn)
                _, _, pool_size = cnn_layer.output_shape

                # construct max pool layer
                pool_layer = lasagne.layers.MaxPool1DLayer(cnn_layer, pool_size=pool_size)
                # reshape the layer to match lstm incoming layer [batch * sent_length, num_filters, 1] --> [batch, sent_length, num_filters]
                output_cnn_layer = lasagne.layers.reshape(pool_layer, (-1, encoderInputs.shape[0], [1]))

                char_params = lasagne.layers.get_all_params(output_cnn_layer, trainable=True)
                self.params += char_params

                char_state_below = lasagne.layers.get_output(output_cnn_layer)


                char_state_below = dropout_layer(char_state_below, use_dropout, trng)

                char_state_shuff = char_state_below.dimshuffle(1,0, 2)
                state_below = tensor.concatenate([state_below, char_state_shuff], axis=2)
                state_below = dropout_layer(state_below, use_dropout, trng)

		for _ in range(self.lstm_layers_num):
			
			enclstm_f = LSTM(emb_size + num_filters, self.en_hidden_size)
			enclstm_b = LSTM(emb_size + num_filters, self.en_hidden_size, True)
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

		state_below = self.de_lookuptable[decoderInputs.flatten()].reshape((decoderInputs.shape[0], decoderInputs.shape[1], self.de_hidden_size))
		for i in range(self.lstm_layers_num):
			declstm = LSTM(self.de_hidden_size, self.de_hidden_size)
			self.decoder_lstm_layers += declstm,    #append
			self.params += declstm.params    #concatenate
			ho, Co = self.hos[i], self.Cos[i]
			state_below, Cs = declstm.forward(state_below, decoderMask, ho, Co)
	
		##### Here we include the representation from the decoder	
		decoder_lstm_outputs = tensor.concatenate([state_below, Encoder], axis=2)

		ei, di, dt = tensor.imatrices(3)    #place holders
		em, dm, tf, di0 =tensor.fmatrices(4) 
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
		loss = costs.sum() / decoderMask.sum(dtype=theano.config.floatX) + params.L2*sum(lasagne.regularization.l2(x) for x in self.params)

		#updates = lasagne.updates.adam(loss, self.params, self.eta)
        	#updates = lasagne.updates.apply_momentum(updates, self.params, momentum=0.9)

		###################################################
		#### using the ground truth when training
		##################################################
		#self._train = theano.function(
		#	inputs=[ei, em, di, dm, dt],
		#	outputs=[loss, softmax_outputs],
		#	updates=updates,
		#	givens={encoderInputs:ei, encoderMask:em, decoderInputs:di, decoderMask:dm, decoderTarget:dt}
		#	)


		#########################################################################
		### For schedule sampling
		#########################################################################
	
		
		###### always use privous predict as next input 
                def _step2(ctx_, state_, hs_, Cs_):

                        hs, Cs = [], []
                        token_idxs = tensor.cast(state_.argmax(axis=-1), "int32" )
			msk_ = tensor.fill( (tensor.zeros_like(token_idxs, dtype="float32")), 1.)
			msk_ = msk_.dimshuffle('x', 0)
                        state_below0 = self.de_lookuptable[token_idxs].reshape((1, encoderInputs.shape[1], self.de_hidden_size))
                        for i, lstm in enumerate(self.decoder_lstm_layers):
                                h, C = lstm.forward(state_below0, msk_, hs_[i], Cs_[i])    #mind msk
                                hs += h[-1],
                                Cs += C[-1],
                                state_below0 = h

                        hs, Cs = tensor.as_tensor_variable(hs), tensor.as_tensor_variable(Cs)
			state_below0 = state_below0.reshape((encoderInputs.shape[1], self.de_hidden_size))			
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
			sequences= [Encoder],
                        outputs_info=[decoderInputs0, hs0, Cs0],
                        n_steps=encoderInputs.shape[0]
                        )
		
		train_predict = train_outputs[0]
		train_costs, _ = theano.scan(fn=_NLL, sequences=[train_predict, decoderTarget, decoderMask])
		
                train_loss = train_costs.sum() / decoderMask.sum(dtype=theano.config.floatX) + params.L2*sum(lasagne.regularization.l2(x) for x in self.params)
		
		##from adam import adam
		##train_updates = adam(train_loss, self.params, self.eta)
                #train_updates = lasagne.updates.sgd(train_loss, self.params, self.eta)
                #train_updates = lasagne.updates.apply_momentum(train_updates, self.params, momentum=0.9)
		from momentum import momentum
                train_updates = momentum(train_loss, self.params, params.eta, momentum=0.9)
		
		self._train2 = theano.function(
                        inputs=[ei, ci, em, di0, dm, dt, use_dropout0],
                        outputs=[train_loss, train_predict],
                        updates=train_updates,
                        givens={encoderInputs:ei, char_input_var:ci, encoderMask:em, decoderInputs0:di0, decoderMask:dm, decoderTarget:dt, use_dropout:use_dropout0}
			#givens={encoderInputs:ei, encoderMask:em, decoderInputs:di, decoderMask:dm, decoderTarget:dt, TF:tf}
                        )
		
		listof_token_idx = train_predict.argmax(axis=-1)
		self._utter = theano.function(
                        inputs=[ei, ci, em, di0, use_dropout0],
                        outputs=listof_token_idx,
                        givens={encoderInputs:ei, char_input_var:ci, encoderMask:em, decoderInputs0:di0, use_dropout:use_dropout0}
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


	def prepare_encoder_data(self, seqs, char_seqs):

                lengths = [len(s) for s in seqs]
                n_samples = len(seqs)
                maxlen = np.max(lengths)


                x = np.zeros((maxlen, n_samples)).astype('int32')
                x_mask = np.zeros((maxlen, n_samples)).astype(theano.config.floatX)
                char_x = np.zeros((n_samples, maxlen, Max_Char_Length)).astype('int32')

                for idx, s in enumerate(seqs):
                        x[:lengths[idx], idx] = s
                        x_mask[:lengths[idx], idx] = 1.
                        for j in range(len(char_seqs[idx])):

                               char_length = len(char_seqs[idx][j])
                               char_x[idx, j, :char_length] = char_seqs[idx][j]

                return x, x_mask, char_x


	def train(self, trainX, devX, testX, params):
		
		self.textfile = open(params.outfile, 'w')

		trainX0, trainX1, trainChar = trainX
                devX0, devX1, devChar = devX
                testX0, testX1, testChar = testX

                devx0, devx0mask, devx0_char = self.prepare_encoder_data(devX0, devChar)
                devx1, _, devy1 = self.prepare_decoder_data(devX1)

                testx0, testx0mask, testx0_char = self.prepare_encoder_data(testX0, testChar)
                testx1, _ , testy1 = self.prepare_decoder_data(testX1)
		
		
		
		bestdev = -1
                best_t =0
             
                try:
                        for eidx in xrange(50):
                                n_samples = 0


                                kf = get_minibatches_idx(len(trainX0), params.batchsize, shuffle=True)
                                uidx = 0
                                aa = 0
                                bb = 0
                                for _, train_index in kf:

                                        uidx += 1

                                        X0 = [trainX0[ii] for ii in train_index]
                                        X1 = [trainX1[ii] for ii in train_index]
                                        XChar = [trainChar[ii] for ii in train_index]

                                        n_samples += len(train_index)

                                        x0, x0mask, x0_char = self.prepare_encoder_data(X0, XChar)
                                        x1_in, _ , x1_out = self.prepare_decoder_data(X1)
					
		
					
					#if (uidx%2==1):
                                        #	traincost, _ = self._train(x0, x0mask, x1_in, x0mask, x1_out)
					#else:
					x1_in2 = np.zeros((len(train_index), self.num_labels+1), dtype = 'float32')
					x1_in2[:,-1] = 1.
					traincost, _ = self._train2(x0, x0_char, x0mask, x1_in2, x0mask, x1_out, 1.)
				       	
					#traincost, _ = self._train(x0, x0mask, x1_in, x0mask, x1_out)
					
 
					#print 'traincost', traincost	
                                        if np.isnan(traincost):
                                                self.textfile.write("NaN detected \n")
                                                self.textfile.flush()



                                devpred  = self.utter(devx0, devx0_char, devx0mask, 0.)
				devacc= 1.0*np.multiply(np.equal(devpred, devy1), devx0mask).sum()/devx0mask.sum()			
			
				                                
                                if bestdev < devacc:
                                        bestdev = devacc
                                        best_t = eidx
                                        #tmp_a_para = [p.get_value() for p in self.params]
                                        #saveParams( tmp_a_para , params.outfile + '.pickle')
					testpred  = self.utter(testx0, testx0_char, testx0mask, 0.)
					testacc= 1.0*np.multiply(np.equal(testpred, testy1), testx0mask).sum()/testx0mask.sum()
				
					self.textfile.write("epoch %d  devacc %f  testacc %f\n" %(eidx , devacc, testacc))
                                        self.textfile.flush()
				
                                	print 'dev acc ', devacc, 'test acc ', testacc

                except KeyboardInterrupt:
                        #print "Classifer Training interupted"
                        self.textfile.write( 'classifer training interrupt \n')
                        self.textfile.flush()
		print 'bestdev', bestdev, 'best_t', best_t
                self.textfile.write("best dev acc: %f  at time %d \n" % (bestdev, best_t))
                self.textfile.close()


		###return self._train(encoderInputs, encoderMask, decoderInputs, decoderMask, decoderTarget)

	def utter(self, encoderInputs, encoderCharInputs, encoderMask, use_dropout):
		n_samples = encoderInputs.shape[1]
		decoderInputs=np.zeros((n_samples, self.num_labels+1),  dtype='float32')
		decoderInputs[:, -1] = 1.
		#if encoderInputs.ndim == 1:
		#	encoderInputs = encoderInputs.reshape((encoderInputs.shape[0], 1))
		#	encoderMask = encoderMask.reshape((encoderMask.shape[0], 1))
		rez = self._utter(encoderInputs, encoderCharInputs, encoderMask, decoderInputs, use_dropout)
		#return rez.reshape((encoderMask.shape[0], encoderMask.shape[1]))
		return rez


	def acc(self, pred, rez, mask):
		pg = np.equal(pred, rez)
		acc = np.sum(pg)
		


