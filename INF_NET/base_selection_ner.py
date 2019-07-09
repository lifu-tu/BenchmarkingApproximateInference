import sys
import warnings
from utils import getGloveWordmap
from params import params
from utils import Get_Ner_bioes
from utils import getTagger
from utils import getTaggerlist
import random
import numpy as np
from base_ner_model_selection import base_model


random.seed(1)
np.random.seed(1)

warnings.filterwarnings("ignore")
params = params()

def Base(eta, l2, inf, hidden_size):
	params.outfile = 'h_base_ner_inf_'
	params.dataf = '../ner_data/eng.train.bioes.conll'
        params.dev = '../ner_data/eng.dev.bioes.conll'
        params.test = '../ner_data/eng.test.bioes.conll'

	params.batchsize = 10
	params.hidden = hidden_size
	params.embedsize = 100
	params.eta = eta
	params.L2 = l2
	params.dropout = 0
	params.emb =0	
	params.inf = inf

	params.en_hidden_size= hidden_size
	params.de_hidden_size= hidden_size
	params.lstm_layers_num =1
	params.num_labels = 17	

	(words, We) = getGloveWordmap('../embedding/glove.6B.100d.txt')
	words.update({'UUUNKKK':0})
	a=[0]*len(We[0])
	newWe = []
	newWe.append(a)
	We = newWe + We
	We = np.asarray(We).astype('float32')
	print We.shape
	tagger = getTagger('../ner_data/ner_bioes')

	params.taggerlist = getTaggerlist('../ner_data/ner_bioes')
	print tagger
	params.outfile = params.outfile+".Batchsize"+'_'+str(params.batchsize)+'_LearningRate_'+str(params.eta)+ '_inf_' +str(inf) + '_' + str(l2) + '_'+ str(hidden_size)
                                #examples are shuffled data
	trainx0, trainy0, _ , _ = Get_Ner_bioes(params.dataf, words, tagger)
        traindata = trainx0, trainy0
        #N = int(params.frac*len(trainx0))
        #traindata = trainx0[:N], trainy0[:N]


        devx0, devy0,  params.devrawx, params.devpos = Get_Ner_bioes(params.dev, words, tagger)
        devdata = devx0, devy0
        print devy0[:10]
        print 'dev set',  len(devx0)
        testx0, testy0, params.testrawx, params.testpos  = Get_Ner_bioes(params.test, words, tagger)
        testdata = testx0, testy0


        print 'test set', len(testx0)
        #print Y
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
                from seq2seq_att_ner_h import Seq2Seq
		#from seq2seq_att_ner_beamsearch import Seq2Seq
		#params.de_hidden_size=200
		#params.outfile = 'de_hidden_200_' + params.outfile
                tm = Seq2Seq(We, params)
                tm.train(traindata, devdata, testdata, params)
	elif(inf ==3):
                #from seq2seq_att_ner import Seq2Seq
                from seq2seq_att_ner_h_beamsearch import Seq2Seq
                #params.de_hidden_size=200
                #params.outfile = 'de_hidden_200_' + params.outfile
                tm = Seq2Seq(We, params)
                tm.train(traindata, devdata, testdata, params)

	elif(inf ==4):
                #from seq2seq_att_all import Seq2Seq
		from seq2seq_local_att_ner import Seq2Seq

		params.window =int(sys.argv[5])
		params.outfile = 'local_att_window_' + str(params.window)+ '_attweight_' +  sys.argv[6] + params.outfile
                tm = Seq2Seq(We, params)
                tm.train(traindata, devdata, testdata, params)
	

if __name__ == "__main__":
       Base(float(sys.argv[1]), float(sys.argv[2]) , int(sys.argv[3]), int(sys.argv[4]))
