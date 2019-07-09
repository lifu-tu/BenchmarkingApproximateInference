import sys
import warnings
from utils import getGloveWordmap
from params import params
from utils import getSupertagData
from utils import getTagger
import random
import numpy as np
from build_CRF_ccctag import CRF_model

random.seed(1)
np.random.seed(1)

warnings.filterwarnings("ignore")
params = params()

def Base(eta, l2, tagversion, hidden , batchsize):
	params.outfile = 'ccctag_CRF_Bilstm_Viterbi_'
	params.dataf = '../supertag_data/train.dat'
	params.dev = '../supertag_data/dev.dat'
	params.test = '../supertag_data/test.dat'
	params.batchsize = batchsize
	params.hidden = hidden
	params.embedsize = 100
	params.emb = 0
	params.eta = eta
	params.L2 = l2
	params.dropout = 0
	params.tagversion = tagversion

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
	elif(tagversion==2):
		tagger = getTagger('../supertag_data/tagger_400')
	else:
		tagger = getTagger('../supertag_data/tagger')

	params.num_labels = len(tagger)
	print len(tagger)
	params.outfile = params.outfile+".Batchsize"+'_'+str(params.batchsize)+'_dropout_'+ str(params.dropout) + "_LearningRate"+'_'+str(params.eta)+ '_' + str(l2)+ str(hidden) + '_tagversoin_'+ str(tagversion)
                                #examples are shuffled data
	
	traindata = getSupertagData(params.dataf, words, tagger)
	trainx0, trainy0 = traindata
 	devdata = getSupertagData(params.dev, words, tagger, train=False)
	devx0, devy0 = devdata
	print 'dev set',  len(devx0)
	testdata = getSupertagData(params.test, words, tagger, train=False)
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
