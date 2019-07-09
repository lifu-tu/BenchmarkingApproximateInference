import sys
import warnings
from utils import getWordmap
from params import params
from utils import getData
#from utils import getUnlabeledData
from utils import getTagger
import random
import numpy as np
from base_model_selection import base_model
#from seq2seq import Seq2Seq
#from seq2seq_att import Seq2Seq
#from seq2seq_att_all import Seq2Seq

random.seed(1)
np.random.seed(1)

warnings.filterwarnings("ignore")
params = params()

def Base(eta, l2, inf, hidden_size):
	params.outfile = 'h_base_pos_inf_'
	params.dataf = '../pos_data/oct27.traindev.proc.cnn'
	params.dev = '../pos_data/oct27.test.proc.cnn'
	params.test = '../pos_data/daily547.proc.cnn'
	params.batchsize = 10
	params.hidden = hidden_size
	params.embedsize = 100
	params.eta = eta
	params.L2 = l2
	params.dropout = 0
	params.emb =0	
	params.inf = inf

	params.en_hidden_size= hidden_size

	"""
	change it later
	"""
	params.de_hidden_size= hidden_size
	params.lstm_layers_num =1
	params.num_labels = 25	

	(words, We) = getWordmap('../embedding/wordvects.tw100w5-m40-it2')
	#words.update({'UUUNKKK':0})
	#a=[0]*len(We[0])
	#newWe = []
	#newWe.append(a)
	#We = newWe + We
	We = np.asarray(We).astype('float32')
	print We.shape
	tagger = getTagger('../pos_data/tagger')
	print tagger
	params.outfile = params.outfile+".Batchsize"+'_'+str(params.batchsize)+'_LearningRate_'+str(params.eta)+ '_inf_' +str(inf) +'_hidden_'+ str(params.hidden)+ '_' + str(l2)
                                #examples are shuffled data
	
	traindata = getData(params.dataf, words, tagger)
	trainx0, trainy0 = traindata
	#N = int(params.frac*len(trainx0))
	#traindata = trainx0[:N], trainy0[:N]
	
 	devdata = getData(params.dev, words, tagger)
	devx0, devy0 = devdata
	print 'dev set',  len(devx0)
	testdata = getData(params.test, words, tagger)
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
                from seq2seq_att_pos_h import Seq2Seq
                tm = Seq2Seq(We, params)
                tm.train(traindata, devdata, testdata, params)
	elif(inf ==3):
                from seq2seq_att_pos_h_beamsearch import Seq2Seq
                tm = Seq2Seq(We, params)
                tm.train(traindata, devdata, testdata, params)


if __name__ == "__main__":
       Base(float(sys.argv[1]), float(sys.argv[2]) , int(sys.argv[3]), int(sys.argv[4]))
