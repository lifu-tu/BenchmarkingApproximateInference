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

def Base(eta, l2, num_filters, inf, hidden_size):
	params.outfile = 'base_ner_inf_'
	params.dataf = '../ner_data/eng.train.bioes.conll'
        params.dev = '../ner_data/eng.dev.bioes.conll'
        params.test = '../ner_data/eng.test.bioes.conll'

	params.batchsize = 10
	params.hidden = hidden_size
	params.embedsize = 100
	params.eta = eta
	params.L2 = l2
	params.dropout = 1
	params.emb =1	
	params.inf = inf

        params.char_embedd_dim =30
        params.num_filters = num_filters
	params.en_hidden_size= hidden_size
	params.de_hidden_size= hidden_size
	params.lstm_layers_num =1
	params.num_labels = 17	
        params.layers_num = 3

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
                             

        char_dic = getTagger('../ner_data/char_dic')
        params.char_dic = char_dic

        scale = np.sqrt(3.0 / params.char_embedd_dim)
        char_embedd_table = np.random.uniform(-scale, scale, [len(char_dic), params.char_embedd_dim]).astype(theano.config.floatX)


        params.taggerlist = getTaggerlist('../ner_data/ner_bioes')
        params.outfile = params.outfile+'_dropout_'+ str(params.dropout) + "_LearningRate"+'_'+str(params.eta)+ '_inf_' + str(inf)+'_' + str(l2)+ '_' + str(num_filters) + '_hidden_'+ str(hidden_size)

        trainx0, trainx0_char, trainy0, _ , _ = Get_Ner_bioes_and_Char(params.dataf, words, tagger, char_dic)
        train = trainx0, trainy0, trainx0_char


        devx0, devx0_char, devy0, params.devrawx, params.devpos = Get_Ner_bioes_and_Char(params.dev, words, tagger, char_dic)
        dev = devx0, devy0, devx0_char
        print devy0[:10]
        print 'dev set',  len(devx0)
        testx0, testx0_char, testy0, params.testrawx, params.testpos  = Get_Ner_bioes_and_Char(params.test, words, tagger, char_dic)
        test = testx0, testy0, testx0_char	
		
	if (inf ==0) or (inf==1):
                from base_ner_model_selection import base_model
		tm = base_model(We, char_embedd_table, params)
		tm.train(train, dev, test, params)
	
	elif(inf ==2):
                from seq2seq_att_ner import Seq2Seq
                tm = Seq2Seq(We, char_embedd_table, params)
                tm.train(train, dev, test, params)

	elif(inf ==3):
                from  self_att_ner import Transformer
                tm = Transformer(We, char_embedd_table, params)
                tm.train(train, dev, test, params)
		

if __name__ == "__main__":
       Base(float(sys.argv[1]), float(sys.argv[2]) , int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
