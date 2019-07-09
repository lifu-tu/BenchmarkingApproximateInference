import sys
import warnings
from utils import getWordmap
from params import params
from utils import getData_and_Char
from utils import getTagger
import random
import numpy as np
from Build_Lstm_Cnn_CRF_POS import CRF_model
import theano
#from build_CRF_C import CRF_model

random.seed(1)
np.random.seed(1)

warnings.filterwarnings("ignore")
params = params()

def Base(eta, l2, num_filters, emb, hidden):
	params.outfile = 'POS_Bilstm_CNN_CRF_'
	params.dataf = '../pos_data/oct27.traindev.proc.cnn'
	params.dev = '../pos_data/oct27.test.proc.cnn'
	params.test = '../pos_data/daily547.proc.cnn'
	params.batchsize = 10
	params.hidden = hidden
	params.embedsize = 100
	params.emb = emb
	params.eta = eta
	params.L2 = l2
	params.dropout = 1
	params.num_labels = 25
        params.char_embedd_dim = 30
        params.num_filters = num_filters    

	(words, We) = getWordmap('../embedding/wordvects.tw100w5-m40-it2')
	#words.update({'UUUNKKK':0})
	#a=[0]*len(We[0])
	#newWe = []
	#newWe.append(a)
	#We = newWe + We
	We = np.asarray(We).astype('float32')
	print We.shape
	tagger = getTagger('../pos_data/tagger')
       
        char_dic = getTagger('../pos_data/char_dic')

        params.char_dic = char_dic

        scale = np.sqrt(3.0 / params.char_embedd_dim)
        char_embedd_table = np.random.uniform(-scale, scale, [len(char_dic), params.char_embedd_dim]).astype(theano.config.floatX)
        
	print char_dic
	params.outfile = params.outfile+".Batchsize"+'_'+str(params.batchsize)+'_dropout_'+ str(params.dropout) + "_LearningRate"+'_'+str(params.eta)+ '_' + str(l2)+ str(num_filters) + '_emb_'+ str(emb)+ '_hidden_'+ str(hidden)
                                #examples are shuffled data
	
	traindata = getData_and_Char(params.dataf, words, tagger, char_dic)
 	devdata = getData_and_Char(params.dev, words, tagger, char_dic)
	testdata = getData_and_Char(params.test, words, tagger, char_dic)


	print 'test set', len(traindata[2])
	#print Y
	print "Using Training Data"+params.dataf
	print "Using Word Embeddings with Dimension "+str(params.embedsize)
	print "Saving models to: "+params.outfile
	#lm = LM_model(params)
	#lm.train(trainy0, devy0, params)	


	tm = CRF_model(We, char_embedd_table, params)
	tm.train(traindata, devdata, testdata, params)

if __name__ == "__main__":
       Base(float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
