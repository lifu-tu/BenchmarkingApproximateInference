import sys
import warnings
from utils import getGloveWordmap
from params import params
from utils import Get_Ner_bioes
#from utils import getUnlabeledData
from utils import getTaggerlist
from utils import getTagger
import random
import numpy as np
from build_CRF_NER import CRF_model
#from build_CRF_C import CRF_model

random.seed(1)
np.random.seed(1)

warnings.filterwarnings("ignore")
params = params()

def Base(eta, l2, morepara, emb , batchsize):
	params.outfile = 'NER_CRF_lstm_Viterti_'
	params.dataf = '../ner_data/eng.train.bioes.conll'
	params.dev = '../ner_data/eng.dev.bioes.conll'
	params.test = '../ner_data/eng.test.bioes.conll'
	params.batchsize = batchsize
	params.hidden = 100
	params.embedsize = 100
	params.emb = emb
	params.eta = eta
	params.L2 = l2
	params.dropout = 0
	params.num_labels = 17

	params.morepara = morepara

	(words, We) = getGloveWordmap('../embedding/glove.6B.100d.txt')
	words.update({'UUUNKKK':0})
	a=[0]*len(We[0])
	newWe = []
	newWe.append(a)
	We = newWe + We
	We = np.asarray(We).astype('float32')
	print We.shape
	tagger = getTagger('../ner_data/ner_bioes')
	print tagger
	params.taggerlist = getTaggerlist('../ner_data/ner_bioes')
	params.outfile = params.outfile+".Batchsize"+'_'+str(params.batchsize)+'_dropout_'+ str(params.dropout) + "_LearningRate"+'_'+str(params.eta)+ '_' + str(l2)+ str(morepara) + '_emb_'+ str(emb)
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
	
	tm = CRF_model(We, params)
	tm.train(trainx0, trainy0, devx0, devy0, testx0, testy0, params)

if __name__ == "__main__":
       Base(float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
