import sys
import warnings
from utils import getGloveWordmap
from params import params
from utils import Get_Ner_bioes_and_Char
from utils import getTaggerlist
from utils import getTagger
import random
import theano
import numpy as np
from build_BiLSTM_CNN_CRF_NER  import CRF_model
#from build_CRF_C import CRF_model

random.seed(1)
np.random.seed(1)

warnings.filterwarnings("ignore")
params = params()

def Base(eta, l2, num_filters, emb , hidden):
	params.outfile = 'NER_BiLSTM_CNN_CRF_'
	params.dataf = '../ner_data/eng.train.bioes.conll'
	params.dev = '../ner_data/eng.dev.bioes.conll'
	params.test = '../ner_data/eng.test.bioes.conll'
	params.batchsize = 10
	params.hidden = hidden
	params.embedsize = 100
	params.emb = emb
	params.eta = eta
	params.L2 = l2
	params.dropout = 1
	params.num_labels = 17
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
	tagger = getTagger('../ner_data/ner_bioes')
	print tagger
        char_dic = getTagger('../ner_data/char_dic')
        params.char_dic = char_dic

        scale = np.sqrt(3.0 / params.char_embedd_dim)
        char_embedd_table = np.random.uniform(-scale, scale, [len(char_dic), params.char_embedd_dim]).astype(theano.config.floatX)


	params.taggerlist = getTaggerlist('../ner_data/ner_bioes')
	params.outfile = params.outfile+".Batchsize"+'_'+str(params.batchsize)+'_dropout_'+ str(params.dropout) + "_LearningRate"+'_'+str(params.eta)+ '_' + str(l2)+ '_' + str(num_filters) + '_hidden_'+ str(hidden)
	
	trainx0, trainx0_char, trainy0, _ , _ = Get_Ner_bioes_and_Char(params.dataf, words, tagger, char_dic)
        train = trainx0, trainy0, trainx0_char


        devx0, devx0_char, devy0, params.devrawx, params.devpos = Get_Ner_bioes_and_Char(params.dev, words, tagger, char_dic)
        dev = devx0, devy0, devx0_char
        print devy0[:10]
        print 'dev set',  len(devx0)
        testx0, testx0_char, testy0, params.testrawx, params.testpos  = Get_Ner_bioes_and_Char(params.test, words, tagger, char_dic)
        test = testx0, testy0, testx0_char


        print 'test set', len(testx0)
        #print Y
        print "Using Training Data"+params.dataf
        print "Using Word Embeddings with Dimension "+str(params.embedsize)
        print "Saving models to: "+params.outfile
	
	tm = CRF_model(We, char_embedd_table, params)
	tm.train(train, dev, test, params)

if __name__ == "__main__":
       Base(float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
