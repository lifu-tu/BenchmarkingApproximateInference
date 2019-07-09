import numpy as np
from params import params
import theano
from theano import tensor as T
import lasagne
import random as random
import pickle
import cPickle
import time
import sys
from lasagne_embedding_layer_2 import lasagne_embedding_layer_2
from random import randint

from crf import CRFLayer
#from crf_utils_copy import crf_loss, crf_accuracy
from crf_utils import crf_loss0, crf_accuracy0






random.seed(1)
np.random.seed(1)
eps = 0.0000001


def saveParams(para, fname):
        f = file(fname, 'wb')
        cPickle.dump(para, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()


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




class CRF_model(object):

	def prepare_data(self, seqs, labels):
		lengths = [len(s) for s in seqs]
                n_samples = len(seqs)
                maxlen = np.max(lengths)
                #sumlen = sum(lengths)

                x = np.zeros((n_samples, maxlen)).astype('int32')
                x_mask = np.zeros((n_samples, maxlen)).astype(theano.config.floatX)
                y = np.zeros((n_samples, maxlen)).astype('int32')

		x_mask1 = np.zeros((n_samples, maxlen)).astype(theano.config.floatX)
                for idx, s in enumerate(seqs):
                        x[idx,:lengths[idx]] = s
                        x_mask[idx,:lengths[idx]] = 1.
                        y[idx,:lengths[idx]] = labels[idx]
			#y[idx,lengths[idx]:] = labels[idx][lengths[idx]-1] 
			x_mask1[idx,lengths[idx]-1] = 1.
                return x, x_mask, x_mask1, y, maxlen
        

		

	def __init__(self,  We_initial, params):
		self.textfile = open(params.outfile, 'w')
		We = theano.shared(We_initial)
        	embsize = We_initial.shape[1]
        	hidden = params.hidden


		trans = np.random.uniform(-0.01, 0.01, (26, 26)).astype('float32')
		transition = theano.shared(trans)


		input_var = T.imatrix(name='inputs')
        	target_var = T.imatrix(name='targets')
        	mask_var = T.fmatrix(name='masks')
		mask_var1 = T.fmatrix(name='masks1')
		length = T.iscalar()


                l_in_word = lasagne.layers.InputLayer((None, None))
                l_mask_word = lasagne.layers.InputLayer(shape=(None, None))

		if params.emb ==1:
                        l_emb_word = lasagne.layers.EmbeddingLayer(l_in_word,  input_size= We_initial.shape[0] , output_size = embsize, W =We)
                else:
                        l_emb_word = lasagne_embedding_layer_2(l_in_word, embsize, We)


		l_lstm_wordf = lasagne.layers.LSTMLayer(l_emb_word, hidden, mask_input=l_mask_word)
        	l_lstm_wordb = lasagne.layers.LSTMLayer(l_emb_word, hidden, mask_input=l_mask_word, backwards = True)

        	concat = lasagne.layers.concat([l_lstm_wordf, l_lstm_wordb], axis=2)
		
		l_reshape_concat = lasagne.layers.ReshapeLayer(concat,(-1,2*hidden))

		l_local = lasagne.layers.DenseLayer(l_reshape_concat, num_units= 25, nonlinearity=lasagne.nonlinearities.linear)


        	#bi_lstm_crf = CRFLayer(concat, params.num_labels, mask_input= l_mask_word)


		local_energy = lasagne.layers.get_output(l_local, {l_in_word: input_var, l_mask_word: mask_var})
		local_energy = local_energy.reshape((-1, length, 25))
                local_energy = local_energy*mask_var[:,:,None]		

		end_term = transition[:-1,-1]
		local_energy = local_energy + end_term.dimshuffle('x', 'x', 0)*mask_var1[:,:, None]

		####local_energy = local_energy*mask_var[:,:,None]		

 		length_index = T.sum(mask_var, axis=1)   

        	loss_train = crf_loss0(local_energy,  transition, target_var, mask_var).mean()
		#loss_train = T.dot(loss_train, length_index)/T.sum(length_index)

		#loss_train = crf_loss0(local_energy, transition, target_var, mask_var).mean()

        	prediction, corr = crf_accuracy0(local_energy, transition, target_var, mask_var)

		##loss_train = crf_loss(energies_train, target_var, mask_var).mean()

                ##prediction, corr = crf_accuracy(energies_train, target_var)


        	corr_train = (corr * mask_var).sum(dtype=theano.config.floatX)
        	num_tokens = mask_var.sum(dtype=theano.config.floatX)



        	network_params = lasagne.layers.get_all_params(l_local, trainable=True)
		network_params.append(transition)

        	print network_params
		self.network_params = network_params

		loss_train = loss_train + params.L2*sum(lasagne.regularization.l2(x) for x in network_params)

        	updates = lasagne.updates.sgd(loss_train, network_params, params.eta)
                updates = lasagne.updates.apply_momentum(updates, network_params, momentum=0.9)

        	self.train_fn = theano.function([input_var, target_var, mask_var, mask_var1, length], [loss_train, corr_train, num_tokens, local_energy], updates=updates, on_unused_input='ignore')

        	self.eval_fn = theano.function([input_var, target_var, mask_var, mask_var1, length], [loss_train, corr_train, num_tokens, prediction], on_unused_input='ignore')




                						

		
	def train(self, trainX, trainY, devX, devY, testX, testY, params):	
		
		trainx0, trainx0mask, trainx0mask1, trainy0, trainmaxlen = self.prepare_data(trainX, trainY)
		#devx0, devx0mask, devx0mask1, devy0, devmaxlen = self.prepare_data(devX[:10], devY[:10])
		#testx0, testx0mask, testx0mask1, testy0, testmaxlen = self.prepare_data(testX[:10], testY[:10])
	
                devx0, devx0mask, devx0mask1, devy0, devmaxlen = self.prepare_data(devX, devY)
                testx0, testx0mask, testx0mask1, testy0, testmaxlen = self.prepare_data(testX, testY)

		start_time = time.time()
        	bestdev = -1

        	bestdev_time =0
        	counter = 0
        	try:
            		for eidx in xrange(30):
                		n_samples = 0

                		start_time1 = time.time()
                		kf = get_minibatches_idx(len(trainX), params.batchsize, shuffle=True)
                		uidx = 0
				aa = 0
				bb = 0
                		for _, train_index in kf:

                    			uidx += 1

                    			x0 = [trainX[ii] for ii in train_index]
                    			y0 = [trainY[ii] for ii in train_index]
                    			n_samples += len(train_index)
					#print y0
					x0, x0mask, x0mask1, y0, maxlen = self.prepare_data(x0, y0)					
					start_time = time.time()
                 			cost, _, _, local_energy = self.train_fn(x0, y0, x0mask, x0mask1 , maxlen)
					end_time = time.time()
					#print('train', end_time-start_time)
					#print cost
					



					#self.textfile.write("hinge loss g:%f    \n" %(cost)  )
					#self.textfile.flush()
			
				
				self.textfile.write("Seen samples:%d   \n" %( n_samples)  )
				self.textfile.flush()
		
				end_time1 = time.time()
				bestdev0 = 0
				best_t0 = 0
				bestdev0_0 = 0
                                best_t1 = 0

								
				### retuning the parameter of student network
				###
				""""
				tmp_d_para = [p.get_value() for p in self.d_params]
                                a =  tmp_d_para[-2]
				print a[1,:]
                                i,j = np.unravel_index(a.argmax(), a.shape)
                                print 'max', i,j
                                i,j = np.unravel_index(a.argmin(), a.shape)
                                print 'min',  i,j
				#b = tmp_d_para[-1]
				#print 'unary term', b
				"""

				start_time2 = time.time()
				
				trainloss, trainpred, trainnum,_    = self.eval_fn(trainx0, trainy0, trainx0mask,trainx0mask1 ,trainmaxlen)
				start_time = time.time()
				devloss, devpred, devnum,devpredtion   = self.eval_fn(devx0, devy0, devx0mask, devx0mask1,devmaxlen)
				end_time = time.time()
				#print('inference', end_time-start_time)
				start_time = time.time()
				testloss, testpred, testnum, _ = self.eval_fn(testx0, testy0, testx0mask, testx0mask1, testmaxlen)
                                end_time = time.time()
				#print('inference', end_time-start_time)
				devacc = 1.0*devpred/devnum
				testacc = 1.0*testpred/testnum
				trainacc = 1.0*trainpred/trainnum
			

				#_, _, _ ,trainpred   = self.eval_fn(x0, y0, x0mask,x0mask1, maxlen)
				#print trainpred	
				print trainloss, devloss, testloss
				print  'prediction', devpredtion[:2] , 'trainacc', trainacc,  'devacc ', devacc, 'testacc ', testacc
				
				end_time2 = time.time()
				if bestdev < devacc:
					bestdev = devacc
					best_t = eidx
					#para = [p.get_value() for p in self.network_params]
					#saveParams(para , params.outfile+ '.pickle')
								
				
					
				#self.textfile.write("epoches %d energy_Cost %f devacc %f devrecall %f devf1 %f testf1 %f trainig time %f test time %f \n" %( eidx + 1, energy_cost ,  best_prec, best_recall,  bestdev0, testf1, end_time1 - start_time1, end_time2 - start_time2 ) )
				self.textfile.write("epoches %d  trainacc %f devacc %f  testacc %f trainig time %f test time %f \n" %( eidx + 1, trainacc, devacc, testacc, end_time1 - start_time1, end_time2 - start_time2 ) )
				self.textfile.flush()
				#self.textfile.write("epoches %d energy_Cost %f devacc %f devrecall %f devf1 %f  training time %f\n" %( eidx + 1, energy_cost ,  best_prec, best_recall,  bestdev0, end_time1 - start_time1) )
			       
        	except KeyboardInterrupt:
            		#print "Classifer Training interupted"
            		self.textfile.write( 'classifer training interrupt \n')
			self.textfile.flush()
        	end_time = time.time()
		#self.textfile.write("total time %f \n" % (end_time - start_time))
		self.textfile.write("best dev acc: %f  at time %d \n" % (bestdev, best_t))
		print 'bestdev ', bestdev, 'at time ',best_t
        	self.textfile.close()
		
