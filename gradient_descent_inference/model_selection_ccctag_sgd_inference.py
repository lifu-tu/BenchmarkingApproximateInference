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






random.seed(1)
np.random.seed(1)
eps = 0.0000001
Max_Char_Length = 30
MAX_lENGTH = 300



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
                
                if (n_samples > 10) or (maxlen > MAX_lENGTH):
                       print 'input sequences are too long or too many samples per minibatch'
                       exit()
                else:
                       ## pad with last examples with 10 minibatch size
                       seqs = seqs + [seqs[-1]]*(10-n_samples)
                       labels = labels + [labels[-1]]*(10-n_samples)
                       char_seqs = char_seqs + [char_seqs[-1]]*(10-n_samples)

                lengths = [len(s) for s in seqs]

                x = np.zeros((10, maxlen)).astype('int32')
                x_mask = np.zeros((10, maxlen)).astype(theano.config.floatX)
                y = np.zeros((10, maxlen)).astype('int32')
                x_mask1 = np.zeros((10, maxlen)).astype(theano.config.floatX)
                char_x = np.zeros((10, maxlen, Max_Char_Length)).astype('int32')

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
		#self.textfile = open(params.outfile, 'w')
		We = theano.shared(We_initial)
               

        	embsize = We_initial.shape[1]
        	hidden = params.hidden

                input_init = np.random.uniform(-0.1, 0.1, (10, MAX_lENGTH, params.num_labels)).astype('float32')
                self.input_init = theano.shared(input_init)

		

		input_var = T.imatrix(name='inputs')
        	target_var = T.imatrix(name='targets')
        	mask_var = T.fmatrix(name='masks')
		mask_var1 = T.fmatrix(name='masks1')
		length = T.iscalar()
			
                                

		Wyy0 = np.random.uniform(-0.02, 0.02, (params.num_labels + 1, params.num_labels)).astype('float32')
                Wyy = theano.shared(Wyy0)


                char_input_var = T.itensor3()

                char_embedd_dim = params.char_embedd_dim
                char_dic_size = len(params.char_dic)
                char_embedd_table = theano.shared(char_embedd_table_initial)
             


                l_in_word = lasagne.layers.InputLayer((None, None))
                l_mask_word = lasagne.layers.InputLayer(shape=(None, None))

		if params.emb ==1:
                        l_emb_word = lasagne.layers.EmbeddingLayer(l_in_word,  input_size= We_initial.shape[0] , output_size = embsize, W =We)
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
                l_emb_word = lasagne.layers.concat([output_cnn_layer, l_emb_word], axis=2)


		l_lstm_wordf = lasagne.layers.LSTMLayer(l_emb_word, hidden, mask_input=l_mask_word)
        	l_lstm_wordb = lasagne.layers.LSTMLayer(l_emb_word, hidden, mask_input=l_mask_word, backwards = True)

        	concat = lasagne.layers.concat([l_lstm_wordf, l_lstm_wordb], axis=2)
		
		l_reshape_concat = lasagne.layers.ReshapeLayer(concat,(-1,2*hidden))

		l_local = lasagne.layers.DenseLayer(l_reshape_concat, num_units= params.num_labels, nonlinearity=lasagne.nonlinearities.linear)

		
		network_params = lasagne.layers.get_all_params(l_local, trainable=True)
                network_params.append(Wyy)

		
		#print len(network_params)
		f = open('ccctag_BiLSTM_CNN_CRF_num_filters_30_dropout_1_LearningRate_0.01_0.0_400_emb_1_tagversoin_2.pickle','r')
		data = pickle.load(f)
		f.close()

		for idx, p in enumerate(network_params):

                        p.set_value(data[idx])

	
	
                
	
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
		local_energy = local_energy.reshape((-1, length, params.num_labels))
                local_energy = local_energy*mask_var[:,:,None]		

		#####################
		# for the end symbole of a sequence
		####################

		end_term = Wyy[:-1,-1]
                local_energy = local_energy + end_term.dimshuffle('x', 'x', 0)*mask_var1[:,:, None]

                
                predy_init = self.input_init[:,:length,:]

                a_params = [self.input_init]


                predy = T.nnet.softmax(predy_init.reshape((-1, params.num_labels)))
                predy = predy.reshape((-1, length, params.num_labels))

                prediction = T.argmax(predy_init, axis=2)

                predy = predy*mask_var[:,:,None]

		
		
		targets_shuffled = predy.dimshuffle(1, 0, 2)
                target_time0 = targets_shuffled[0]
		
		masks_shuffled = mask_var.dimshuffle(1, 0)		 

                initial_energy0 = T.dot(target_time0, Wyy[-1,:-1])


                initials = [target_time0, initial_energy0]
                [ _, target_energies], _ = theano.scan(fn=inner_function, outputs_info=initials, sequences=[targets_shuffled[1:], masks_shuffled[1:]])
                cost11 = target_energies[-1] + T.sum(T.sum(local_energy*predy, axis=2)*mask_var, axis=1)
				
		
		predy_f =  predy.reshape((-1, params.num_labels))
		y_f = target_var.flatten()

	
		if (params.annealing ==0):
                        lamb = params.L3
                elif (params.annealing ==1):
                        lamb = params.L3* (1 - 0.01*t_t)


		cost = T.mean(-cost11)
                		   
                
		#from adam import adam
                #updates_a = adam(cost, a_params, params.eta)
					
		updates_a = lasagne.updates.sgd(cost, a_params, params.eta)
                updates_a = lasagne.updates.apply_momentum(updates_a, a_params, momentum=0.9)

	        self.inf_fn = theano.function([input_var, char_input_var, mask_var, mask_var1, length], cost, updates = updates_a)
                self.eval_fn = theano.function([input_var, char_input_var, mask_var, mask_var1, length], [prediction, -cost11], on_unused_input='ignore')	
                						
                if params.WarmStart:

                                       We_inf = theano.shared(We_initial)
                                       hidden_inf= params.hidden_inf
                                       char_embedd_table_inf = theano.shared(char_embedd_table_initial)


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

                                       l_lstm_wordf_a = lasagne.layers.LSTMLayer(l_emb_word_a, hidden_inf, mask_input=l_mask_word_a)
                                       l_lstm_wordb_a = lasagne.layers.LSTMLayer(l_emb_word_a, hidden_inf, mask_input=l_mask_word_a, backwards = True)

                                       l_reshapef_a = lasagne.layers.ReshapeLayer(l_lstm_wordf_a ,(-1, hidden_inf))
                                       l_reshapeb_a = lasagne.layers.ReshapeLayer(l_lstm_wordb_a ,(-1,hidden_inf))
                                       concat2_a = lasagne.layers.ConcatLayer([l_reshapef_a, l_reshapeb_a])

                                       l_local_a = lasagne.layers.DenseLayer(concat2_a, num_units= params.num_labels, nonlinearity=lasagne.nonlinearities.linear)

                                       predy_inf = lasagne.layers.get_output(l_local_a, {l_in_word_a:input_var, l_mask_word_a:mask_var, layer_char_input_a:char_input_var})
                                       predy_inf = predy_inf.reshape((-1, length, params.num_labels))

                                       a_params = lasagne.layers.get_all_params(l_local_a, trainable=True)

                                       f = open('CRF_Inf_ccctag.Batchsize_10_dropout_1_LearningRate_0.005_1.0_num_filters_30_inf_0_hidden_400_tagversion_2.pickle','r')
                                       data = pickle.load(f)
                                       f.close()


                                       for idx, p in enumerate(a_params):
                                               p.set_value(data[idx])

                                       self.start_fn = theano.function([input_var, char_input_var, mask_var, length], predy_inf, on_unused_input='ignore')


		
	def train(self, train, dev, test, params):	
		
                                trainX, trainY, trainChar = train
                                devX, devY, devChar = dev
                                testX, testY, testChar = test

		
		
		                tagger_keys = params.tagger.keys()
                                tagger_values = params.tagger.values()

                                words_keys = params.words.keys()
                                words_values = params.words.values()
		
     
              
                                kf = get_minibatches_idx(len(devX), params.batchsize, shuffle=False)
                         
                                C = 0
                                N = 0
                                E = 0
                                test_time = []
                                for _, dev_index in kf:

                               

                                        x0 = [devX[ii] for ii in dev_index]
                                        y0 = [devY[ii] for ii in dev_index]
                                        charx0 = [devChar[ii] for ii in dev_index]

                                        n_samples = len(dev_index)

                                        
                                        x0, x0mask, x0_char, x0mask1, y0, maxlen = self.prepare_data(x0, y0, charx0)

                                        predy_init = np.random.uniform(-0.1, 0.1, (10, MAX_lENGTH, params.num_labels)).astype('float32')

                                        if params.WarmStart:
                                                 predy_init0 = self.start_fn(x0, x0_char, x0mask, maxlen)

                                                 predy_init[:,:maxlen,:] = predy_init0
      
                                        self.input_init.set_value(predy_init)
                                        start_time = time.time()
                                        for i0 in range(params.epoches):
                                               cost = self.inf_fn(x0, x0_char, x0mask, x0mask1, maxlen)
                                               #print cost

                                        end_time = time.time()

                                        test_time.append(end_time-start_time)

                                        pred0, cost0 = self.eval_fn(x0, x0_char, x0mask, x0mask1, maxlen)

                                        corr = np.sum(np.equal(pred0[:n_samples,:], y0[:n_samples,:])*x0mask[:n_samples,:])
                                        num = np.sum(x0mask[:n_samples,:])

                                        C += corr
                                        N += num
                                        E += np.sum(cost0[:n_samples])

                                print 'inference time', np.mean(test_time), E/len(devX)
                                print 'devacc', 1.0*C/N



                                kf = get_minibatches_idx(len(testX), params.batchsize, shuffle=False)

                                C = 0
                                N = 0
                                E = 0
                                test_time = []
                                for _, test_index in kf:

                       

                                        x0 = [testX[ii] for ii in test_index]
                                        y0 = [testY[ii] for ii in test_index]
                                        charx0 = [testChar[ii] for ii in test_index]

                                        n_samples = len(test_index)


                                        x0, x0mask, x0_char, x0mask1, y0, maxlen = self.prepare_data(x0, y0, charx0)

                                        predy_init = np.random.uniform(-0.1, 0.1, (10, MAX_lENGTH, params.num_labels)).astype('float32')

                                        if params.WarmStart:
                                                 predy_init0 = self.start_fn(x0, x0_char, x0mask, maxlen)

                                                 predy_init[:,:maxlen,:] = predy_init0

                                        self.input_init.set_value(predy_init)

                                        start_time = time.time()
                                        for i0 in range(params.epoches):
                                               cost = self.inf_fn(x0, x0_char, x0mask, x0mask1, maxlen)
                                               #print cost
                                        end_time = time.time()

                                        test_time.append(end_time-start_time)

                                        pred0, cost0 = self.eval_fn(x0, x0_char, x0mask, x0mask1, maxlen)

                                        corr = np.sum(np.equal(pred0[:n_samples,:], y0[:n_samples,:])*x0mask[:n_samples,:])
                                        num = np.sum(x0mask[:n_samples,:])

                                        C += corr
                                        N += num
                                        E += np.sum(cost0[:n_samples])

                                print 'inference time', np.sum(test_time)          
                                print 'testacc', 1.0*C/N, E/len(testX)


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
					
			
