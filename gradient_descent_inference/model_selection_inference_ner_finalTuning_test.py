import sys
import warnings
from utils import getGloveWordmap
from params import params
from utils import Get_Ner_bioes_and_Char
from utils import getTagger
from utils import getTaggerlist

import random
import numpy as np
import theano

random.seed(1)
np.random.seed(1)

warnings.filterwarnings("ignore")
params = params()

def Base(eta, epoches):
	params.outfile = 'CRF_Inf_NER_'
	params.dataf = '../ner_data/eng.train.bioes.conll'
        params.dev = '../ner_data/eng.dev.bioes.conll'
        params.test = '../ner_data/eng.test.bioes.conll'
	
        hidden_inf = 200
        emb =1
	params.batchsize = 10
        params.hidden = 200
        params.embedsize = 100
        params.emb = emb
        params.eta = eta
        params.dropout = 1
	params.hidden_inf = hidden_inf
        params.de_hidden_size = 200

        params.char_embedd_dim = 30

        num_filters = 50
        params.num_filters = num_filters


        params.epoches = epoches
        

      
        params.regutype = 0
        params.annealing = 0
     
	

	(words, We) = getGloveWordmap('../embedding/glove.6B.100d.txt')
	words.update({'UUUNKKK':0})
	a=[0]*len(We[0])
	newWe = []
	newWe.append(a)
	We = newWe + We
	We = np.asarray(We).astype('float32')
	tagger = getTagger('../ner_data/ner_bioes')
	params.taggerlist = getTaggerlist('../ner_data/ner_bioes')	

        char_dic = getTagger('../ner_data/char_dic')
        params.char_dic = char_dic

        scale = np.sqrt(3.0 / params.char_embedd_dim)
        char_embedd_table = np.random.uniform(-scale, scale, [len(char_dic), params.char_embedd_dim]).astype(theano.config.floatX)
 
	params.words = words
	params.tagger = tagger
       
	params.outfile = params.outfile+'_LearningRate_'+str(params.eta)+ '_epoch_'+ str(epoches)
	
        print params.outfile

        trainx0, trainx0_char, trainy0, _ , _ = Get_Ner_bioes_and_Char(params.dataf, words, tagger, char_dic)
        train = trainx0, trainy0, trainx0_char

        devx0, devx0_char, devy0, params.devrawx, params.devpos = Get_Ner_bioes_and_Char(params.dev, words, tagger, char_dic)
        dev = devx0, devy0, devx0_char
        
        testx0, testx0_char, testy0, params.testrawx, params.testpos  = Get_Ner_bioes_and_Char(params.test, words, tagger, char_dic)
        test = testx0, testy0, testx0_char



        #from model_selection_NER_finalTuning_sgd_inference import CRF_model
        #tm = CRF_model(We, char_embedd_table, params)
        from model_selection_NER_seq2seq_finalTuning_sgd_inference import CRF_seq2seq_model
	tm = CRF_seq2seq_model(We, char_embedd_table, params)
	tm.train(train, dev, test, params)

	
if __name__ == "__main__":
	Base(float(sys.argv[1]), int(sys.argv[2]))
