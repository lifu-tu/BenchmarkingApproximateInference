import sys
import warnings
from utils import getWordmap
from params import params
from utils import getData_and_Char
from utils import getTagger
import random
import numpy as np
import theano

random.seed(1)
np.random.seed(1)

warnings.filterwarnings("ignore")
params = params()


def Base(eta, l3, emb, num_filters, inf, hidden_inf):
	params.outfile = 'Pos_CRF_Inf_'
	params.dataf = '../pos_data/oct27.traindev.proc.cnn'
	params.dev = '../pos_data/oct27.test.proc.cnn'
	params.test = '../pos_data/daily547.proc.cnn'

        params.char_embedd_dim = 30
        params.num_filters = num_filters
	params.batchsize = 10
        params.hidden = 100
        params.embedsize = 100
        params.emb = emb
        params.eta = eta
        params.dropout = 1
	params.hidden_inf = hidden_inf
	params.small = 0


        params.inf = inf
        params.regutype = 0
        params.annealing = 0
        params.L3 = l3
	

	(words, We) = getWordmap('../embedding/wordvects.tw100w5-m40-it2')
	We = np.asarray(We).astype('float32')
	print We.shape
	tagger = getTagger('../pos_data/tagger')
	print tagger
	params.words = words
	params.tagger = tagger
   
        params.num_labels = len(tagger) 
   
        char_dic = getTagger('../pos_data/char_dic')

        params.char_dic = char_dic

        scale = np.sqrt(3.0 / params.char_embedd_dim)
        char_embedd_table = np.random.uniform(-scale, scale, [len(char_dic), params.char_embedd_dim]).astype(theano.config.floatX)

		
	params.outfile = params.outfile+".num_filters"+'_'+str(num_filters)+'_dropout_'+ str(params.dropout) + '_LearningRate_'+str(params.eta)+ '_'  + str(l3) +'_emb_'+ str(emb)+ '_inf_'+ str(params.inf)+ '_hidden_inf_'+ str(params.hidden_inf)

        train = getData_and_Char(params.dataf, words, tagger, char_dic)
        dev = getData_and_Char(params.dev, words, tagger, char_dic)
        test = getData_and_Char(params.test, words, tagger, char_dic)
	

	if (inf ==0) or (inf==1):
                from model_selection_inference import CRF_model
		tm = CRF_model(We, char_embedd_table, params)
		tm.train(train, dev, test, params)

	elif(inf==2):
		from model_selection_inference_seq2seq import CRF_seq2seq_model

		params.de_hidden_size = hidden_inf
		tm = CRF_seq2seq_model(We, char_embedd_table, params)
                tm.train(train, dev, test, params)

	elif(inf==3):
                from model_selection_inference_seq2seq_beamsearch import CRF_seq2seq_model
                params.de_hidden_size = hidden_inf
                tm = CRF_seq2seq_model(We, char_embedd_table, params)
                tm.train(train, dev, test, params)



if __name__ == "__main__":
	Base(float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
