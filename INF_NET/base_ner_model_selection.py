import theano
import numpy as np
from theano import tensor as T
from theano import config
import random as random
import time
import utils
from lasagne_embedding_layer_2 import lasagne_embedding_layer_2 


import lasagne
import sys
import cPickle
import pickle

import subprocess
from subprocess import Popen, PIPE, STDOUT


class base_model(object):

    #takes list of seqs, puts them in a matrix
    #returns matrix of seqs and mask	
    def prepare_data(self, seqs, labels):

        lengths = [len(s) for s in seqs]
    	n_samples = len(seqs)
    	maxlen = np.max(lengths)
	sumlen = sum(lengths)
	
	y =  [item for a in labels for item in a ]
    	x = np.zeros((n_samples, maxlen)).astype('int32')
	#x = np.ones((n_samples, maxlen)).astype('int32')
    	y = np.asarray(y).astype('int32')
    	x_mask = np.zeros((n_samples, maxlen)).astype(theano.config.floatX)
    	for idx, s in enumerate(seqs):
        	x[idx,:lengths[idx]] = s
        	x_mask[idx,:lengths[idx]] = 1.

    	return x, x_mask, y

    def get_idxs(self, xmask):
        tmp = xmask.reshape(-1,1)
        idxs = []
        for i in range(len(tmp)):
            if tmp[i] > 0:
                idxs.append(i)
        return np.asarray(idxs).astype('int32')    


    def saveParams(self, para, fname):
        f = file(fname, 'wb')
        cPickle.dump(para, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    def get_minibatches_idx(self, n, minibatch_size, shuffle=False):
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


    def __init__(self,   We_initial,  params):

       
     
	self.eta = params.eta
        We = theano.shared(We_initial)
	embsize = We_initial.shape[1]
	hidden = params.hidden

	g = T.imatrix()
        gmask = T.fmatrix()
        y = T.ivector()
	idxs = T.ivector()


        l_in_word = lasagne.layers.InputLayer((None, None))
        l_mask_word = lasagne.layers.InputLayer(shape=(None, None))

	if params.emb ==1:
                        l_emb_word = lasagne.layers.EmbeddingLayer(l_in_word,  input_size= We_initial.shape[0] , output_size = embsize, W =We)
        else:
                        l_emb_word = lasagne_embedding_layer_2(l_in_word, embsize, We)
        #l_emb_word = lasagne.layers.EmbeddingLayer(l_in_word,  input_size= We_initial.shape[0]   , output_size = embsize , W =We)
	#l_emb_word = lasagne_embedding_layer_2(l_in_word,  embsize , We)
	
    	if params.dropout:
        	l_emb_word = lasagne.layers.DropoutLayer(l_emb_word, p=0.5)
	
	
	if (params.inf==0):
                        l_lstm_wordf = lasagne.layers.LSTMLayer(l_emb_word, hidden, mask_input=l_mask_word)
                        l_lstm_wordb = lasagne.layers.LSTMLayer(l_emb_word, hidden, mask_input=l_mask_word, backwards = True)

                        l_reshapef = lasagne.layers.ReshapeLayer(l_lstm_wordf,(-1,hidden))
                        l_reshapeb = lasagne.layers.ReshapeLayer(l_lstm_wordb,(-1,hidden))
                        concat2 = lasagne.layers.ConcatLayer([l_reshapef, l_reshapeb])
        elif(params.inf==1) :
                        l_cnn_input = lasagne.layers.DimshuffleLayer(l_emb_word, (0, 2, 1))
                        l_cnn_1 = lasagne.layers.Conv1DLayer(l_cnn_input, hidden, 1, 1, pad = 'same')
                        l_cnn_3 = lasagne.layers.Conv1DLayer(l_cnn_input, hidden, 3, 1, pad = 'same')
			l_cnn = lasagne.layers.ConcatLayer([l_cnn_1, l_cnn_3], axis=1)			
			#l_cnn = lasagne.layers.Conv1DLayer(l_cnn_input, hidden, 1, 1, pad = 'same')
                        concat2 = lasagne.layers.DimshuffleLayer(l_cnn, (0, 2, 1))
                        #concat2 = lasagne.layers.ConcatLayer([l_emb_word, concat2], axis =2)
                        concat2 = lasagne.layers.ReshapeLayer(concat2 ,(-1, 2*hidden))
	else:
			l_cnn_input = lasagne.layers.DimshuffleLayer(l_emb_word, (0, 2, 1))
                        l_cnn = lasagne.layers.Conv1DLayer(l_cnn_input, hidden, 3, 1, pad = 'same')
			concat2 = lasagne.layers.DimshuffleLayer(l_cnn, (0, 2, 1))
			concat2 = lasagne.layers.ReshapeLayer(concat2 ,(-1, hidden))
			concat2 = lasagne.layers.DenseLayer(concat2, num_units= hidden)

	
    	if params.dropout:
        	concat2 = lasagne.layers.DropoutLayer(concat2, p=0.5)

        #l_emb = lasagne.layers.DenseLayer(concat2, num_units=hidden, nonlinearity=lasagne.nonlinearities.tanh)
        l_out = lasagne.layers.DenseLayer(concat2, num_units= params.num_labels, nonlinearity=lasagne.nonlinearities.softmax)


	output = lasagne.layers.get_output(l_out, {l_in_word: g, l_mask_word: gmask})
	
	output_1= output[idxs]

	test_output = lasagne.layers.get_output(l_out, {l_in_word: g, l_mask_word: gmask}, deterministic=True)

        test_output_1= test_output[idxs]

	model_params = lasagne.layers.get_all_params(l_out, trainable=True)
	self.model_p = lasagne.layers.get_all_params(l_out, trainable=True)

	reg = sum(lasagne.regularization.l2(x) for x in model_params)

	cost = lasagne.objectives.categorical_crossentropy(output_1, y)	
	cost = T.mean(cost) + params.L2 * reg

	#pred = T.argmax(output_1, axis=1)
	final_pred = T.argmax(test_output_1, axis=1)

 	y1 = T.ones_like(y)
        SUM = T.sum(y1)
        acc = 1.0 * T.sum(T.eq(final_pred, y))/SUM

	self.acc_function = theano.function([g, gmask, y, idxs], [acc, final_pred], on_unused_input='warn')	

	#updates = lasagne.updates.adam(cost, model_params, self.eta)
	#from adam import adam
        #updates = adam(cost, model_params, self.eta)
	updates = lasagne.updates.sgd(cost, model_params, self.eta)
        updates = lasagne.updates.apply_momentum(updates, model_params, momentum=0.9)
	self.train_function = theano.function([g, gmask, y, idxs], [cost, acc], updates=updates, on_unused_input='warn')	




    def train(self, traindata, devdata, testdata, params):

	self.textfile = open(params.outfile, 'w')

	trainx0, trainy0 = traindata
        devx, devy = devdata
        testx, testy = testdata

        devx0, devx0mask, devy0 = self.prepare_data(devx, devy)
        devidx = self.get_idxs(devx0mask)


        testx0, testx0mask, testy0 = self.prepare_data(testx, testy)
        testidx = self.get_idxs(testx0mask)

	start_time = time.time()
	bestdev =0
        bestdev_time =0
	counter = 0

        try:
            for eidx in xrange(50):
                n_samples = 0

                # Get new shuffled index for the training set.
                kf = self.get_minibatches_idx(len(trainx0), params.batchsize, shuffle=True)
                uidx = 0
                for _, train_index in kf:

                    uidx += 1

                    x0 = [trainx0[t] for t in train_index]
                    y0 = [trainy0[t] for t in train_index]
                    n_samples += len(train_index)
                    x0, x0mask, y0 = self.prepare_data(x0, y0)
		    idx0 = self.get_idxs(x0mask)
		   
		    start_time = time.time()    
                    traincost, trainacc = self.train_function(x0, x0mask, y0, idx0)
		    end_time = time.time()
		    ##print(end_time-start_time)
		start_time = time.time()
                
		devcost, dev_pred = self.acc_function(devx0, devx0mask, devy0, devidx)

		end_time = time.time()
		
		f = open('pred_txt_'+ params.outfile , 'w')
                devlength = [len(s) for s in devx]
                aaaa = 0
                for ii, devl in enumerate(devlength):
                        for jj in range(devl):
                                        eva_string = params.devrawx[ii][jj] + ' '+ params.devpos[ii][jj] + ' '+ params.taggerlist[devy0[aaaa]] +' ' +  params.taggerlist[dev_pred[aaaa]]
                                        #eva_string =  'John NNP ' + params.taggerlist[devy0[aaaa]] +' ' +  params.taggerlist[dev_pred[aaaa]]
                                        #if len(eva_string.split(' '))!=4:
                                        #       print   len(eva_string.split(' '))  , eva_string
                                        f.write(eva_string + '\n'  )
                                        aaaa +=1
                        f.write('\n')
                f.close()
                #subprocess.call("perl conlleval < pred.txt", shell=True)
                cmd = 'perl conlleval < pred_txt_' + params.outfile
                p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
                output = p.stdout.read()
                output =  output.split('\n')
                output = output[1]
                output = output.split(' ')
                dev_f1 = float(output[-1])

                ##print('inference', end_time-start_time)
                start_time = time.time()
                testacc = self.acc_function(testx0, testx0mask, testy0, testidx)
                end_time = time.time()
                ##print('inference', end_time-start_time)
        
                 
	        if dev_f1 > bestdev:
                        bestdev = dev_f1
                        bestdev_time = eidx + 1	  
			#para = [p.get_value() for p in self.model_p]
			#self.saveParams(para , 'F0_new.pickle')
			testcost, test_pred = self.acc_function(testx0, testx0mask, testy0, testidx)
			f = open('pred_txt_' + params.outfile, 'w')
                	testlength = [len(s) for s in testx]
                	aaaa = 0
                	for ii, testl in enumerate(testlength):
                        	for jj in range(testl):
                                	eva_string = params.testrawx[ii][jj] + ' '+ params.testpos[ii][jj] + ' '+ params.taggerlist[testy0[aaaa]] +' ' +  params.taggerlist[test_pred[aaaa]]
                                	f.write(eva_string + '\n'  )
                                	aaaa +=1
                        	f.write('\n')
                	f.close()
                	cmd = 'perl conlleval < pred_txt_' + params.outfile
                	p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
                	output = p.stdout.read()
                	output =  output.split('\n')
                	output = output[1]
                	output = output.split(' ')
                	test_f1 = float(output[-1])
			print 'devf ', dev_f1, 'testf ', test_f1
			self.textfile.write("epoch %d  devf1 %f  testf1 %f\n" %(eidx , dev_f1, test_f1))
                        self.textfile.flush()

			self.saveParams(dev_pred, params.outfile + '_predy_dev.pickle')
                        self.saveParams(test_pred, params.outfile + '_predy_test.pickle')
          

                if np.isnan(traincost) or np.isinf(traincost):
                        print 'NaN detected'



                #if(params.save):
                 #   counter += 1
                  #  self.saveParams(params.outfile+ 'Classifier'+str(counter)+'.pickle')

                #evaluate_all(self,self.words)

                print 'Seen %d samples' % n_samples


        except KeyboardInterrupt:
            print "Tagger Training interupted"
	end_time = time.time()
	self.textfile.write("best dev acc: %f  at time %d \n" % (bestdev, bestdev_time))
        self.textfile.close()
	print "best dev acc:", bestdev, ' at time:  ', bestdev_time    
