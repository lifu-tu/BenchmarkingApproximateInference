import sys
import warnings
from utils import getGloveWordmap
from params import params
from utils import getSupertagData_and_Char
from utils import getTagger
import random
import numpy as np
from Build_Lstm_Cnn_CRF_ccctag import CRF_model
import theano

random.seed(1)
np.random.seed(1)

warnings.filterwarnings("ignore")
params = params()

def Base(eta, l2, tagversion, hidden, num_filters):
	params.outfile = 'ccctag_BiLSTM_CNN_CRF_'
	params.dataf = '../supertag_data/train.dat'
	params.dev = '../supertag_data/dev.dat'
	params.test = '../supertag_data/test.dat'
	params.batchsize = 10
	params.hidden = hidden
	params.embedsize = 100
	params.emb = 1
	params.eta = eta
	params.L2 = l2
	params.dropout = 1
	params.tagversion = tagversion
        params.char_embedd_dim = 30
        params.num_filters = num_filters
        


	(words, We) = getGloveWordmap('../embedding/glove.6B.100d.txt')
	words.update({'UUUNKKK':0})
	a=[0]*len(We[0])
	newWe = []
	newWe.append(a)
	We = newWe + We
	We = np.asarray(We).astype('float32')
	print We.shape
	if (tagversion==0):
		tagger = getTagger('../supertag_data/tagger_100')
	elif(tagversion==1):
		tagger = getTagger('../supertag_data/tagger_200')
	else:
		tagger = getTagger('../supertag_data/tagger_400')
	params.num_labels = len(tagger)
	print len(tagger)
      
        char_dic = getTagger('../supertag_data/char_dic')

        params.char_dic = char_dic

        scale = np.sqrt(3.0 / params.char_embedd_dim)
        char_embedd_table = np.random.uniform(-scale, scale, [len(char_dic), params.char_embedd_dim]).astype(theano.config.floatX)


	params.outfile = params.outfile+"num_filters_"+str(num_filters)+'_dropout_'+ str(params.dropout) + "_LearningRate"+'_'+str(params.eta)+ '_' + str(l2)+ '_' + str(hidden) + '_emb_'+ str(params.emb) +  '_tagversoin_'+ str(tagversion)
            
	


	train = getSupertagData_and_Char(params.dataf, words, tagger, char_dic)

 	dev = getSupertagData_and_Char(params.dev, words, tagger, char_dic, train=False)

	test = getSupertagData_and_Char(params.test, words, tagger, char_dic, train=False)	

	#print Y
	print "Using Training Data"+params.dataf
	print "Using Word Embeddings with Dimension "+str(params.embedsize)
	print "Saving models to: "+params.outfile


	tm = CRF_model(We, char_embedd_table, params)
	tm.train(train, dev, test, params)

if __name__ == "__main__":
       Base(float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
