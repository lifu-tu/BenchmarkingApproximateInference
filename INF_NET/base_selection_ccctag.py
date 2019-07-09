import sys
import warnings
from utils import getGloveWordmap
from params import params
from utils import getSupertagData
#from utils import getUnlabeledData
from utils import getTagger
import random
import numpy as np
from base_model_selection_ccctag import base_model
#from seq2seq import Seq2Seq
#from seq2seq_att import Seq2Seq
#from seq2seq_att_all import Seq2Seq

random.seed(1)
np.random.seed(1)

warnings.filterwarnings("ignore")
params = params()

def Base(eta, l2, inf, tagversion, hidden):
	params.outfile = 'h_base_ccctag_inf_'
	params.dataf = '../supertag_data/train.dat'
	params.dev = '../supertag_data/dev.dat'
	params.test = '../supertag_data/test.dat'
	params.batchsize = 10
	params.hidden = hidden
	params.embedsize = 100
	params.eta = eta
	params.L2 = l2
	params.dropout = 0
	params.emb =0	
	params.inf = inf
	
	params.en_hidden_size = hidden
	params.de_hidden_size= hidden
	params.lstm_layers_num =1
	

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
	
	params.outfile = params.outfile+".Batchsize"+'_'+str(params.batchsize)+'_LearningRate_'+str(params.eta)+ '_inf_' +str(inf) +'_hidden_'+ str(params.hidden)+ '_' + str(tagversion)+ '_'+ str(l2)
                                #examples are shuffled data
	
	traindata = getSupertagData(params.dataf, words, tagger)
        trainx0, trainy0 = traindata
        devdata = getSupertagData(params.dev, words, tagger, train=False)
        devx0, devy0 = devdata
        print 'dev set',  len(devx0)
        testdata = getSupertagData(params.test, words, tagger, train=False)
        testx0, testy0 = testdata

	print 'test set', len(testx0)
	print "Using Training Data"+params.dataf
	print "Using Word Embeddings with Dimension "+str(params.embedsize)
	print "Saving models to: "+params.outfile
	
	if (inf ==0) or (inf==1):
		tm = base_model(We, params)
		tm.train(traindata, devdata, testdata, params)
	#elif(inf ==2):
	#	from seq2seq import Seq2Seq
	#	tm = Seq2Seq(We, params)
	#	tm.train(traindata, devdata, testdata, params)
	elif(inf ==2):
                #from seq2seq_att_pos import Seq2Seq
                from seq2seq_att_pos_new import Seq2Seq
                tm = Seq2Seq(We, params)
                tm.train(traindata, devdata, testdata, params)
	elif(inf ==3):
                ##from seq2seq_att_pos_beamsearch import Seq2Seq
                from seq2seq_att_pos_new_beamsearch import Seq2Seq
                tm = Seq2Seq(We, params)
                tm.train(traindata, devdata, testdata, params)
	#elif(inf ==3):
        #        from seq2seq_att_all import Seq2Seq
        #        tm = Seq2Seq(We, params)
        #        tm.train(traindata, devdata, testdata, params)


if __name__ == "__main__":
       Base(float(sys.argv[1]), float(sys.argv[2]) , int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
