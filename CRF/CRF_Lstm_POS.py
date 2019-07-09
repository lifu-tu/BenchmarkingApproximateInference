import sys
import warnings
from utils import getWordmap
from params import params
from utils import getData
#from utils import getUnlabeledData
from utils import getTagger
import random
import numpy as np
from build_CRF_POS import CRF_model
#from build_CRF_C import CRF_model

random.seed(1)
np.random.seed(1)

warnings.filterwarnings("ignore")
params = params()

def Base(eta, l2, morepara, emb , batchsize):
	params.outfile = 'POS_CRF_Bilstm_Viterbi_'
	params.dataf = '../pos_data/oct27.traindev.proc.cnn'
	params.dev = '../pos_data/oct27.test.proc.cnn'
	params.test = '../pos_data/daily547.proc.cnn'
	params.batchsize = batchsize
	params.hidden = 100
	params.embedsize = 100
	params.emb = emb
	params.eta = eta
	params.L2 = l2
	params.dropout = 0
	params.num_labels = 25

	params.morepara = morepara

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
	params.outfile = params.outfile+".Batchsize"+'_'+str(params.batchsize)+'_dropout_'+ str(params.dropout) + "_LearningRate"+'_'+str(params.eta)+ '_' + str(l2)+ str(morepara) + '_emb_'+ str(emb)
                                #examples are shuffled data
	
	traindata = getData(params.dataf, words, tagger)
	trainx0, trainy0 = traindata
 	devdata = getData(params.dev, words, tagger)
	devx0, devy0 = devdata
	print 'dev set',  len(devx0)
	testdata = getData(params.test, words, tagger)
	testx0, testy0 = testdata	

	print 'test set', len(testx0)
	#print Y
	print "Using Training Data"+params.dataf
	print "Using Word Embeddings with Dimension "+str(params.embedsize)
	print "Saving models to: "+params.outfile
	#lm = LM_model(params)
	#lm.train(trainy0, devy0, params)	


	tm = CRF_model(We, params)
	tm.train(trainx0, trainy0, devx0, devy0, testx0, testy0, params)

if __name__ == "__main__":
       Base(float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
