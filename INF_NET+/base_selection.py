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

def Base(eta, l2, num_filters, inf, hidden_size):
	params.outfile = 'base_pos_inf_'
	params.dataf = '../pos_data/oct27.traindev.proc.cnn'
	params.dev = '../pos_data/oct27.test.proc.cnn'
	params.test = '../pos_data/daily547.proc.cnn'
	params.batchsize = 10
	params.hidden = hidden_size
	params.embedsize = 100
	params.eta = eta
	params.L2 = l2
	params.dropout = 1
	params.emb = 1	
	params.inf = inf

        params.char_embedd_dim = 30
        params.num_filters = num_filters
	params.en_hidden_size= hidden_size

	"""
	change it later
	"""
	params.de_hidden_size= hidden_size
	params.lstm_layers_num =1
	params.num_labels = 25	
        params.layers_num = 3

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

        char_dic = getTagger('../pos_data/char_dic')

        params.char_dic = char_dic

        scale = np.sqrt(3.0 / params.char_embedd_dim)
        char_embedd_table = np.random.uniform(-scale, scale, [len(char_dic), params.char_embedd_dim]).astype(theano.config.floatX)

	params.outfile = params.outfile+".num_filters"+'_'+str(num_filters)+'_LearningRate_'+str(params.eta)+ '_inf_' +str(inf) +'_hidden_'+ str(params.hidden)+ '_' + str(l2)      


        train = getData_and_Char(params.dataf, words, tagger, char_dic)
        dev = getData_and_Char(params.dev, words, tagger, char_dic)
        test = getData_and_Char(params.test, words, tagger, char_dic)
	
	if (inf ==0) or (inf==1):
                from base_model_selection import base_model
		tm = base_model(We, char_embedd_table, params)
		tm.train(train, dev, test, params)
	
	elif(inf ==2):
                from seq2seq_att_pos import Seq2Seq
                tm = Seq2Seq(We, char_embedd_table, params)
                tm.train(train, dev, test, params)

	elif(inf ==3):
                from self_att import Transformer
                tm = Transformer(We, char_embedd_table, params)
                tm.train(train, dev, test, params)


if __name__ == "__main__":
       Base(float(sys.argv[1]), float(sys.argv[2]) , int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
