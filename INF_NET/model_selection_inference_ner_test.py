import sys
import warnings
from utils import getGloveWordmap
from params import params
from utils import Get_Ner_bioes
from utils import getTagger
from utils import getTaggerlist

import random
import numpy as np

random.seed(1)
np.random.seed(1)

warnings.filterwarnings("ignore")
params = params()

def Base(eta, l3, emb, batchsize, inf, hidden_inf):
	params.outfile = 'h_CRF_Inf_NER_'
	params.dataf = '../ner_data/eng.train.bioes.conll'
        params.dev = '../ner_data/eng.dev.bioes.conll'
        params.test = '../ner_data/eng.test.bioes.conll'
	
	params.batchsize = batchsize
        params.hidden = 100
        params.embedsize = 100
        params.emb = emb
        params.eta = eta
        params.dropout = 0
	params.hidden_inf = hidden_inf


        params.inf = inf
        params.regutype = 0
        params.annealing = 0
        params.L3 = l3
	

	(words, We) = getGloveWordmap('../embedding/glove.6B.100d.txt')
	words.update({'UUUNKKK':0})
	a=[0]*len(We[0])
	newWe = []
	newWe.append(a)
	We = newWe + We
	We = np.asarray(We).astype('float32')
	tagger = getTagger('../ner_data/ner_bioes')
	params.taggerlist = getTaggerlist('../ner_data/ner_bioes')	

	params.words = words
	params.tagger = tagger
       
	params.outfile = params.outfile+".Batchsize"+'_'+str(params.batchsize)+'_dropout_'+ str(params.dropout) + '_LearningRate_'+str(params.eta)+ '_'  + str(l3) +'_emb_'+ str(emb)+ '_inf_'+ str(params.inf)+ '_hidden_'+ str(params.hidden_inf)
	

	trainx0, trainy0, _ , _ = Get_Ner_bioes(params.dataf, words, tagger)
        traindata = trainx0, trainy0
        

        devx0, devy0,  params.devrawx, params.devpos = Get_Ner_bioes(params.dev, words, tagger)
        devdata = devx0, devy0
        print devy0[:10]
        print 'dev set',  len(devx0)
        testx0, testy0, params.testrawx, params.testpos  = Get_Ner_bioes(params.test, words, tagger)
        testdata = testx0, testy0

	print "Using Training Data"+params.dataf
	print "Using Word Embeddings with Dimension "+str(params.embedsize)
	print "Saving models to: "+params.outfile
	#lm = LM_model(params)
	#lm.train(trainy0, devy0, params)	

	if (inf ==0) or (inf==1):
		from model_selection_NER_inference import CRF_model
		tm = CRF_model(We, params)
		tm.train(trainx0, trainy0, devx0, devy0, testx0, testy0, params)
	elif(inf==2):
		from model_selection_inference_NER_seq2seq_h import CRF_seq2seq_model
		params.de_hidden_size = hidden_inf
		params.outfile = 'h_de_hidden_' + str(params.de_hidden_size) + '_' + params.outfile
		tm = CRF_seq2seq_model(We, params)
                tm.train(trainx0, trainy0, devx0, devy0, testx0, testy0, params)

	else:
                from model_selection_inference_NER_seq2seq_h_beamsearch import CRF_seq2seq_model
                params.de_hidden_size = hidden_inf
                params.outfile = 'h_de_hidden_' + str(params.de_hidden_size) + '_' + params.outfile
                tm = CRF_seq2seq_model(We, params)
                tm.train(trainx0, trainy0, devx0, devy0, testx0, testy0, params)

if __name__ == "__main__":
	Base(float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
