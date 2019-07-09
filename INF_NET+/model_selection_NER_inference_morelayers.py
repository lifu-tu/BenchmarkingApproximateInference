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

from crf_utils import crf_loss0, crf_accuracy0

import subprocess
from subprocess import Popen, PIPE, STDOUT




random.seed(1)
np.random.seed(1)
eps = 0.0000001
Max_Char_Length = 30

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

	def prepare_data(self, seqs, labels, char_seqs):
                lengths = [len(s) for s in seqs]
                n_samples = len(seqs)
                maxlen = np.max(lengths)

                x = np.zeros((n_samples, maxlen)).astype('int32')
                x_mask = np.zeros((n_samples, maxlen)).astype(theano.config.floatX)
                y = np.zeros((n_samples, maxlen)).astype('int32')
                x_mask1 = np.zeros((n_samples, maxlen)).astype(theano.config.floatX)
                char_x = np.zeros((n_samples, maxlen, Max_Char_Length)).astype('int32')

                for idx, s in enumerate(seqs):
                        x[idx,:lengths[idx]] = s
                        x_mask[idx,:lengths[idx]] = 1.
                        y[idx,:lengths[idx]] = labels[idx]
                        x_mask1[idx,lengths[idx]-1] = 1.
                        for j in range(len(char_seqs[idx])):

                               char_length = len(char_seqs[idx][j])
                               char_x[idx, j, :char_length] = char_seqs[idx][j]

                return x, x_mask, char_x, x_mask1, y, maxlen        

		

	def __init__(self,  We_initial, char_embedd_table_initial, params):
		self.textfile = open(params.outfile, 'w')
		We = theano.shared(We_initial)
                We_inf = theano.shared(We_initial)
        	embsize = We_initial.shape[1]
        	hidden = params.hidden

		hidden_inf= params.hidden_inf
		

		input_var = T.imatrix(name='inputs')
        	target_var = T.imatrix(name='targets')
        	mask_var = T.fmatrix(name='masks')
		mask_var1 = T.fmatrix(name='masks1')
		length = T.iscalar()
		t_t = T.fscalar()		

		Wyy0 = np.random.uniform(-0.02, 0.02, (18, 18)).astype('float32')
                Wyy = theano.shared(Wyy0)

                char_input_var = T.itensor3()

                char_embedd_dim = params.char_embedd_dim
                char_dic_size = len(params.char_dic)
                char_embedd_table = theano.shared(char_embedd_table_initial)
                char_embedd_table_inf = theano.shared(char_embedd_table_initial)

                l_in_word = lasagne.layers.InputLayer((None, None))
                l_mask_word = lasagne.layers.InputLayer(shape=(None, None))

		if params.emb ==1:
                        l_emb_word = lasagne.layers.EmbeddingLayer(l_in_word,  input_size= We_initial.shape[0] , output_size = embsize, W =We, name='word_embedding')
                else:
                        l_emb_word = lasagne_embedding_layer_2(l_in_word, embsize, We)

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
                output_cnn_layer = lasagne.layers.reshape(pool_layer, (-1, length, [1]))

                # finally, concatenate the two incoming layers together.
                incoming = lasagne.layers.concat([output_cnn_layer, l_emb_word], axis=2)


		l_lstm_wordf = lasagne.layers.LSTMLayer(incoming, hidden, mask_input=l_mask_word)
        	l_lstm_wordb = lasagne.layers.LSTMLayer(incoming, hidden, mask_input=l_mask_word, backwards = True)

        	concat = lasagne.layers.concat([l_lstm_wordf, l_lstm_wordb], axis=2)
		
		l_reshape_concat = lasagne.layers.ReshapeLayer(concat,(-1,2*hidden))

		l_local = lasagne.layers.DenseLayer(l_reshape_concat, num_units= 17, nonlinearity=lasagne.nonlinearities.linear)

		
		network_params = lasagne.layers.get_all_params(l_local, trainable=True)
                network_params.append(Wyy)

		
		print len(network_params)
		f = open('NER_BiLSTM_CNN_CRF_.Batchsize_10_dropout_1_LearningRate_0.005_0.0_50_hidden_200.pickle','r')
		data = pickle.load(f)
		f.close()

		for idx, p in enumerate(network_params):

                        p.set_value(data[idx])

	
		l_in_word_a = lasagne.layers.InputLayer((None, None))
                l_mask_word_a = lasagne.layers.InputLayer(shape=(None, None))

                l_emb_word_a = lasagne.layers.EmbeddingLayer(l_in_word_a,  input_size= We_initial.shape[0] , output_size = embsize, W = We_inf, name='inf_word_embedding')


                layer_char_input_a = lasagne.layers.InputLayer(shape=(None, None, Max_Char_Length ),
                                                     input_var=char_input_var, name='char-input')

                layer_char_a = lasagne.layers.reshape(layer_char_input_a, (-1, [2]))
                layer_char_embedding_a = lasagne.layers.EmbeddingLayer(layer_char_a, input_size=char_dic_size,
                                                             output_size=char_embedd_dim, W=char_embedd_table_inf,
                                                             name='char_embedding')


                layer_char_a = lasagne.layers.DimshuffleLayer(layer_char_embedding_a, pattern=(0, 2, 1))


                # first get some necessary dimensions or parameters
                conv_window = 3
                num_filters = params.num_filters
                #_, sent_length, _ = incoming2.output_shape

                # dropout before cnn?
                if params.dropout:
                     layer_char_a = lasagne.layers.DropoutLayer(layer_char_a, p=0.5)

                # construct convolution layer
                cnn_layer_a = lasagne.layers.Conv1DLayer(layer_char_a, num_filters=num_filters, filter_size=conv_window, pad='full',
                                           nonlinearity=lasagne.nonlinearities.tanh, name='cnn')
                # infer the pool size for pooling (pool size should go through all time step of cnn)
                #_, _, pool_size = cnn_layer.output_shape

                # construct max pool layer
                pool_layer_a = lasagne.layers.MaxPool1DLayer(cnn_layer_a, pool_size=pool_size)
                # reshape the layer to match lstm incoming layer [batch * sent_length, num_filters, 1] --> [batch, sent_length, num_filters]
                output_cnn_layer_a = lasagne.layers.reshape(pool_layer_a, (-1, length, [1]))             
                 
                # finally, concatenate the two incoming layers together.
                l_emb_word_a = lasagne.layers.concat([output_cnn_layer_a, l_emb_word_a], axis=2)

                if params.dropout:
                     l_emb_word_a = lasagne.layers.DropoutLayer(l_emb_word_a, p=0.5)

		if (params.inf ==0):
                        l_lstm_wordf_a = lasagne.layers.LSTMLayer(l_emb_word_a, hidden_inf, mask_input=l_mask_word_a)
                        l_lstm_wordb_a = lasagne.layers.LSTMLayer(l_emb_word_a, hidden_inf, mask_input=l_mask_word_a, backwards = True)
                                                 
                        l_emb_word_a1 = lasagne.layers.concat([l_lstm_wordf_a, l_lstm_wordb_a], axis=2)

                        l_lstm_wordf_a = lasagne.layers.LSTMLayer(l_emb_word_a1, hidden_inf, mask_input=l_mask_word_a)
                        l_lstm_wordb_a = lasagne.layers.LSTMLayer(l_emb_word_a1, hidden_inf, mask_input=l_mask_word_a, backwards = True)


                        l_reshapef_a = lasagne.layers.ReshapeLayer(l_lstm_wordf_a ,(-1, hidden_inf))
                        l_reshapeb_a = lasagne.layers.ReshapeLayer(l_lstm_wordb_a ,(-1, hidden_inf))
                        concat2_a = lasagne.layers.ConcatLayer([l_reshapef_a, l_reshapeb_a])


                else:
                        l_cnn_input_a = lasagne.layers.DimshuffleLayer(l_emb_word_a, (0, 2, 1))
                        l_cnn_3_a = lasagne.layers.Conv1DLayer(l_cnn_input_a, hidden_inf, 3, 1, pad = 'same')
			l_cnn_1_a = lasagne.layers.Conv1DLayer(l_cnn_input_a, hidden_inf, 1, 1, pad = 'same')
                        #l_cnn_a = lasagne.layers.Conv1DLayer(l_cnn_a, hidden, 3, 1, pad = 'same')
			l_cnn_a = lasagne.layers.ConcatLayer([l_cnn_1_a, l_cnn_3_a], axis=1)
                        concat2_a = lasagne.layers.DimshuffleLayer(l_cnn_a, (0, 2, 1))
                        concat2_a = lasagne.layers.ReshapeLayer(concat2_a ,(-1, 2*hidden_inf))

		
        	if params.dropout:
                        concat2_a = lasagne.layers.DropoutLayer(concat2_a, p=0.5)

                l_local_a = lasagne.layers.DenseLayer(concat2_a, num_units= 17, nonlinearity=lasagne.nonlinearities.softmax)

		a_params = lasagne.layers.get_all_params(l_local_a, trainable=True)
                self.a_params = a_params

		def inner_function( targets_one_step, mask_one_step,  prev_label, tg_energy):
                        """
                        :param targets_one_step: [batch_size, t]
                        :param prev_label: [batch_size, t]
                        :param tg_energy: [batch_size]
                        :return:
                        """                 
                        new_ta_energy = T.dot(prev_label, Wyy[:-1,:-1])
                        new_ta_energy_t = tg_energy + T.sum(new_ta_energy*targets_one_step, axis =1)
			tg_energy_t = T.switch(mask_one_step, new_ta_energy_t,  tg_energy)

                        return [targets_one_step, tg_energy_t]


		local_energy = lasagne.layers.get_output(l_local, {l_in_word: input_var, l_mask_word: mask_var, layer_char_input:char_input_var})
		local_energy = local_energy.reshape((-1, length, 17))
                local_energy = local_energy*mask_var[:,:,None]		

		#####################
		# for the end symbole of a sequence
		####################

		end_term = Wyy[:-1,-1]
                local_energy = local_energy + end_term.dimshuffle('x', 'x', 0)*mask_var1[:,:, None]


		predy0 = lasagne.layers.get_output(l_local_a, {l_in_word_a:input_var, l_mask_word_a:mask_var, layer_char_input_a:char_input_var})

                predy_inf = lasagne.layers.get_output(l_local_a, {l_in_word_a:input_var, l_mask_word_a:mask_var, layer_char_input_a:char_input_var}, deterministic=True)
                predy_inf = predy_inf.reshape((-1, length, 17))               
 

		predy_in = T.argmax(predy0, axis=1)
                A = T.extra_ops.to_one_hot(predy_in, 17)
                A = A.reshape((-1, length, 17))		

		predy = predy0.reshape((-1, length, 17))
		predy = predy*mask_var[:,:,None]

		
		targets_shuffled = predy.dimshuffle(1, 0, 2)
                target_time0 = targets_shuffled[0]
		
		masks_shuffled = mask_var.dimshuffle(1, 0)		 

                initial_energy0 = T.dot(target_time0, Wyy[-1,:-1])


                initials = [target_time0, initial_energy0]
                [ _, target_energies], _ = theano.scan(fn=inner_function, outputs_info=initials, sequences=[targets_shuffled[1:], masks_shuffled[1:]])
                cost11 = target_energies[-1] + T.sum(T.sum(local_energy*predy, axis=2)*mask_var, axis=1)

		
		# compute the ground-truth energy

		targets_shuffled0 = A.dimshuffle(1, 0, 2)
                target_time00 = targets_shuffled0[0]


                initial_energy00 = T.dot(target_time00, Wyy[-1,:-1])


                initials0 = [target_time00, initial_energy00]
                [ _, target_energies0], _ = theano.scan(fn=inner_function, outputs_info=initials0, sequences=[targets_shuffled0[1:], masks_shuffled[1:]])
                cost110 = target_energies0[-1] + T.sum(T.sum(local_energy*A, axis=2)*mask_var, axis=1)
		
		
		predy_f =  predy.reshape((-1, 17))
		y_f = target_var.flatten()

	
		if (params.annealing ==0):
                        lamb = params.L3
                elif (params.annealing ==1):
                        lamb = params.L3*(np.e)**(-0.01*t_t) 


		if (params.regutype==0):
                        ce_hinge = lasagne.objectives.categorical_crossentropy(predy_f + eps, y_f)
                        ce_hinge = ce_hinge.reshape((-1, length))
                        ce_hinge = T.sum(ce_hinge* mask_var, axis=1)
			cost = T.mean(-cost11) + lamb*T.mean(ce_hinge)
                else:

                        entropy_term = - T.sum(predy_f * T.log(predy_f + eps), axis=1)
                        entropy_term = entropy_term.reshape((-1, length))
                        entropy_term = T.sum(entropy_term*mask_var, axis=1)
			cost = T.mean(-cost11) - lamb*T.mean(entropy_term)
                    
		###from adam import adam
                ###updates_a = adam(cost, a_params, params.eta)
					
		updates_a = lasagne.updates.sgd(cost, a_params, params.eta)
                updates_a = lasagne.updates.apply_momentum(updates_a, a_params, momentum=0.9)

		if (params.regutype==0):
			self.train_fn = theano.function([input_var, char_input_var, target_var, mask_var, mask_var1, length, t_t], [cost, ce_hinge], updates = updates_a, on_unused_input='ignore')
		else:
			self.train_fn = theano.function([input_var, char_input_var, target_var, mask_var, mask_var1, length, t_t], [cost, entropy_term], updates = updates_a, on_unused_input='ignore')


		prediction = T.argmax(predy_inf, axis=2)
		corr = T.eq(prediction, target_var)
        	corr_train = (corr * mask_var).sum(dtype=theano.config.floatX)
        	num_tokens = mask_var.sum(dtype=theano.config.floatX)
        	

        	self.eval_fn = theano.function([input_var, char_input_var, target_var, mask_var, mask_var1, length], [corr_train, num_tokens, prediction], on_unused_input='ignore')




                						

		
	def train(self, train, dev, test, params):	
	
                trainX, trainY, trainChar = train
                devX, devY, devChar = dev
                testX, testY, testChar = test
	
                devx0, devx0mask, devx0_char, devx0mask1, devy0, devmaxlen = self.prepare_data(devX, devY, devChar)
                testx0, testx0mask, testx0_char, testx0mask1, testy0, testmaxlen = self.prepare_data(testX, testY, testChar)	
		
		start_time = time.time()
        	bestdev = -1
		
		tagger_keys = params.tagger.keys()
                tagger_values = params.tagger.values()

                words_keys = params.words.keys()
                words_values = params.words.values()
		
        	best_t =0
        	counter = 0
        	try:
            		for eidx in xrange(80):
                		n_samples = 0

                		start_time1 = time.time()
                		kf = get_minibatches_idx(len(trainX), params.batchsize, shuffle=True)
                		uidx = 0
				#aa = 0
				#bb = 0
                		for _, train_index in kf:

                    			uidx += 1

                    			x0 = [trainX[ii] for ii in train_index]
                    			y0 = [trainY[ii] for ii in train_index]
                                        charx0 = [trainChar[ii] for ii in train_index]
                    			n_samples += len(train_index)
					#print y0
					x0, x0mask, x0_char, x0mask1, y0, maxlen = self.prepare_data(x0, y0, charx0)					
					
                 			cost, regu_cost = self.train_fn(x0, x0_char, y0, x0mask, x0mask1, maxlen, eidx)
					#print regu_cost
								
				
				self.textfile.write("Seen samples:%d   \n" %( n_samples)  )
				self.textfile.flush()
		
				end_time1 = time.time()

								
				
				start_time2 = time.time()
				
				#trainloss, trainpred, trainnum,_    = self.eval_fn(trainx0, trainy0, trainx0mask, trainmaxlen)
				devpred, devnum, dev_pred   = self.eval_fn(devx0, devx0_char, devy0, devx0mask, devx0mask1, devmaxlen)
				#testloss, _, testpred, testnum, test_pred = self.eval_fn(testx0, testy0, testx0mask, testx0mask1, testmaxlen)
				
		
				f = open('pred_crf_lstm_txt_' + params.outfile, 'w')
                                devlength = [len(s) for s in devX]
                                aaaa = 0
                                for ii, devl in enumerate(devlength):
                                        for jj in range(devl):
                                                eva_string = params.devrawx[ii][jj] + ' '+ params.devpos[ii][jj] + ' '+ params.taggerlist[devy0[ii,jj]] +' ' +  params.taggerlist[dev_pred[ii,jj]]
                                                #eva_string =  'John NNP ' + params.taggerlist[devy0[aaaa]] +' ' +  params.taggerlist[dev_pred[aaaa]]
                                                if len(eva_string.split(' '))!=4:
                                                        print   len(eva_string.split(' '))  , eva_string
                                                f.write(eva_string + '\n'  )
                                                aaaa +=1
                                        f.write('\n')
                                f.close()

                                cmd = 'perl conlleval < pred_crf_lstm_txt_' + params.outfile
                                p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
                                output = p.stdout.read()
                                output =  output.split('\n')
                                output = output[1]
                                output = output.split(' ')
                                dev_f10 = float(output[-1])
			

			
				testlength = [len(s) for s in testX]	
				#print devloss, testloss
				#print  'devacc ', devacc
				
				end_time2 = time.time()
				if bestdev < dev_f10:
					bestdev = dev_f10
					best_t = eidx
					
					testpred, testnum, test_pred = self.eval_fn(testx0, testx0_char, testy0, testx0mask, testx0mask1, testmaxlen)
					f = open('pred_crf_lstm_txt_' + params.outfile, 'w')
                                	testlength = [len(s) for s in testX]
                                	for ii, testl in enumerate(testlength):
                                        	for jj in range(testl):
                                                	eva_string = params.testrawx[ii][jj] + ' '+ params.testpos[ii][jj] + ' '+ params.taggerlist[testy0[ii,jj]] +' ' +  params.taggerlist[test_pred[ii,jj]]

                                                	f.write(eva_string + '\n'  )
                                        	f.write('\n')
                                	f.close()
                                	cmd = 'perl conlleval < pred_crf_lstm_txt_' + params.outfile
                                	p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
                                	output = p.stdout.read()
                                	output =  output.split('\n')
                                	output = output[1]
                                	output = output.split(' ')
                                	test_f10 = float(output[-1])
                                	#print 'epoch ', (eidx),  'test f1', test_f10
                                	self.textfile.write("epoch %d devf1 %f   testf1 %f\n" %(eidx , dev_f10, test_f10))
                                	self.textfile.flush()	
					
					#saveParams(dev_pred, params.outfile + '_predy_dev.pickle')
                                        #saveParams(test_pred, params.outfile + '_predy_test.pickle')			

					#para = [p.get_value() for p in self.a_params]
					#saveParams(para , params.outfile+ '.pickle')
					"""
					f = open('QA_' + params.outfile, 'w')
                                        for i in range(testloss.shape[0]):
                                                tweet_i = testX[i]
                                                pred_i = test_pred[i].tolist()
                                                pred_i = pred_i[:testlength[i]]
                                                pred_i = [ tagger_keys[tagger_values.index(x)] for x in pred_i]
                                                pred_i = ' '.join(str(x) for x in pred_i)
                                                tweet_i = [words_keys[words_values.index(x)] for x in tweet_i]
                                                tweet_i = ' '.join(str(x) for x in tweet_i)

                                                tweet_y_i = testY[i]
                                                y_i = [ tagger_keys[tagger_values.index(x)] for x in tweet_y_i]
                                                y_i = ' '.join(str(x) for x in y_i)

                                                f.write("%s |||\t%s\t|||%s\n" %(tweet_i, pred_i, y_i))
                                        f.close()
					"""

					"""
					saveParams(dev_pred, 'predy_dev.pickle')

                                        f = open('Loss_' + params.outfile, 'w')
                                        for i in range(devloss.shape[0]):
                                                pred_i = dev_pred[i].tolist()
                                                pred_i = pred_i[:devlength[i]]
                                                pred_i = ' '.join(str(x) for x in pred_i)
                                                f.write("%f \t %s\n" %(devloss[i], pred_i))
                                        f.close()

                                        f = open('Loss0_' + params.outfile, 'w')
                                        for i in range(devloss0.shape[0]):
                                                pred_i = dev_pred[i].tolist()
                                                pred_i = pred_i[:devlength[i]]
                                                pred_i = ' '.join(str(x) for x in pred_i)
                                                f.write("%f \t %s\n" %(devloss0[i], pred_i))
                                        f.close()			
					"""
					
				
					#self.textfile.write("epoches %d  devacc %f  testacc %f trainig time %f test time %f \n" %( eidx + 1, dev_f10, test_f10, end_time1 - start_time1, end_time2 - start_time2 ) )
					#self.textfile.flush()
				
			       
        	except KeyboardInterrupt:
            		self.textfile.write( 'classifer training interrupt \n')
			self.textfile.flush()
        	end_time = time.time()
		self.textfile.write("best dev f1: %f  at time %d \n" % (bestdev, best_t))
		#print 'bestdev ', bestdev, 'at time ',best_t
        	self.textfile.close()
		
