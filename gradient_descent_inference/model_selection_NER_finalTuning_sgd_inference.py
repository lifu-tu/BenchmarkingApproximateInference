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


                l_lstm_wordf_a = lasagne.layers.LSTMLayer(l_emb_word_a, hidden_inf, mask_input=l_mask_word_a)
                l_lstm_wordb_a = lasagne.layers.LSTMLayer(l_emb_word_a, hidden_inf, mask_input=l_mask_word_a, backwards = True)

                l_reshapef_a = lasagne.layers.ReshapeLayer(l_lstm_wordf_a ,(-1, hidden_inf))
                l_reshapeb_a = lasagne.layers.ReshapeLayer(l_lstm_wordb_a ,(-1,hidden_inf))
                concat2_a = lasagne.layers.ConcatLayer([l_reshapef_a, l_reshapeb_a])
                
		
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
				
		
		predy_f =  predy.reshape((-1, 17))


                entropy_term = - T.sum(predy_f * T.log(predy_f + eps), axis=1)
                entropy_term = entropy_term.reshape((-1, length))
                entropy_term = T.sum(entropy_term*mask_var, axis=1)
		cost = T.mean(-cost11)
               



                ### compute the energy for inference step
                predy_inf = predy_inf*mask_var[:,:,None]

                targets_inf_shuffled = predy_inf.dimshuffle(1, 0, 2)
                target_inf_time0 = targets_inf_shuffled[0]

                initial_inf_energy0 = T.dot(target_inf_time0, Wyy[-1,:-1])


                initials_inf = [target_inf_time0, initial_inf_energy0]
                [ _, target_inf_energies], _ = theano.scan(fn=inner_function, outputs_info=initials_inf, sequences=[targets_inf_shuffled[1:], masks_shuffled[1:]])
                cost_inf = target_inf_energies[-1] + T.sum(T.sum(local_energy*predy_inf, axis=2)*mask_var, axis=1)

     
		###from adam import adam
                ###updates_a = adam(cost, a_params, params.eta)
					
		updates_a = lasagne.updates.sgd(cost, a_params, params.eta)
                updates_a = lasagne.updates.apply_momentum(updates_a, a_params, momentum=0.9)

	
		self.train_fn = theano.function([input_var, char_input_var, mask_var, mask_var1, length], cost, updates = updates_a, on_unused_input='ignore')

		


		prediction = T.argmax(predy_inf, axis=2)
        	

        	self.eval_fn = theano.function([input_var, char_input_var, mask_var, mask_var1, length], [prediction, -cost_inf], on_unused_input='ignore')




                						

		
	def train(self, train, dev, test, params):	
	
                                trainX, trainY, trainChar = train
                                devX, devY, devChar = dev
                                testX, testY, testChar = test
	
                	
		
		                tagger_keys = params.tagger.keys()
                                tagger_values = params.tagger.values()

                                words_keys = params.words.keys()
                                words_values = params.words.values()
		
     
        

                		f = open('CRF_Inf_NER_.num_filters_50_dropout_1_LearningRate_0.001_1.0_emb_1_inf_0_hidden_200_annealing_0.pickle','r')
                                data = pickle.load(f)
                                f.close()
								
				
				start_time2 = time.time()
				
                                kf = get_minibatches_idx(len(devX), params.batchsize, shuffle=False)

                                dev_pred = []
                                n_samples = 0
                                E1 = 0

                                test_time = []
                                for _, dev_index in kf:

                                        x0 = [devX[ii] for ii in dev_index]
                                        y0 = [devY[ii] for ii in dev_index]
                                        charx0 = [devChar[ii] for ii in dev_index]
                                        n_samples = len(dev_index)
                                        length0 = [len(devX[ii]) for ii in dev_index]

                                        x0, x0mask, x0_char, x0mask1, y0, maxlen = self.prepare_data(x0, y0, charx0)
                                     
                                        for idx, p in enumerate(self.a_params):
                                               p.set_value(data[idx])

                                        start_time = time.time()

                                        for i0 in range(params.epoches):
                                                     cost = self.train_fn(x0, x0_char, x0mask, x0mask1, maxlen)
                                        end_time = time.time()
                                        test_time.append(end_time-start_time)

                                        dev_pred0, cost0 = self.eval_fn(x0, x0_char, x0mask, x0mask1, maxlen)
         
                                        E1 +=np.sum(cost0[:n_samples])

                                        for i in range(len(dev_index)):
                                                dev_pred.append(dev_pred0[i,:length0[i]].tolist())                                        

                                print np.mean(test_time)
                                        
			
				
		
				f = open('pred_crf_lstm_txt_' + params.outfile, 'w')
                                devlength = [len(s) for s in devX]
                                aaaa = 0
                                for ii, devl in enumerate(devlength):
                                        for jj in range(devl):
                                                eva_string = params.devrawx[ii][jj] + ' '+ params.devpos[ii][jj] + ' '+ params.taggerlist[devY[ii][jj]] +' ' +  params.taggerlist[dev_pred[ii][jj]]
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
			

                                kf = get_minibatches_idx(len(testX), params.batchsize, shuffle=False)

                                test_pred = []
                                n_samples = 0
                                E2 = 0
                                test_time = []
                                for _, test_index in kf:

                                        x0 = [testX[ii] for ii in test_index]
                                        y0 = [testY[ii] for ii in test_index]
                                        charx0 = [testChar[ii] for ii in test_index]
                                        n_samples = len(test_index)
                                        length0 = [len(testX[ii]) for ii in test_index]

                                        x0, x0mask, x0_char, x0mask1, y0, maxlen = self.prepare_data(x0, y0, charx0)

                                        for idx, p in enumerate(self.a_params):
                                               p.set_value(data[idx])

                                        start_time = time.time()

                                        for i0 in range(params.epoches):
                                                     cost = self.train_fn(x0, x0_char, x0mask, x0mask1, maxlen)
                                        end_time = time.time()
                                        test_time.append(end_time-start_time)

                                        test_pred0, cost0 = self.eval_fn(x0, x0_char, x0mask, x0mask1, maxlen)
                                        E2 +=np.sum(cost0[:n_samples])

                                        for i in range(len(test_index)):
                                                test_pred.append(test_pred0[i,:length0[i]].tolist())

                                print np.mean(test_time)

			
				
				f = open('pred_crf_lstm_txt_' + params.outfile, 'w')
                                testlength = [len(s) for s in testX]
                                for ii, testl in enumerate(testlength):
                                        	for jj in range(testl):
                                                	eva_string = params.testrawx[ii][jj] + ' '+ params.testpos[ii][jj] + ' '+ params.taggerlist[testY[ii][jj]] +' ' +  params.taggerlist[test_pred[ii][jj]]

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
                                print 'devf1 ', dev_f10, E1/len(devX), 'test f1', test_f10, E2/len(testX)
                              	
					
	
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
					
		
