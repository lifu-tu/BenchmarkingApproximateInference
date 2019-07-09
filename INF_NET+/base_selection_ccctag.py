import sys
import warnings
from utils import getGloveWordmap
from params import params
from utils import getSupertagData_and_Char
from utils import getTagger
import random
import numpy as np

random.seed(1)
np.random.seed(1)
import theano

warnings.filterwarnings("ignore")
params = params()

def Base(eta, l2, num_filters, inf, tagversion, hidden):
	params.outfile = 'base_ccctag_inf_'
	params.dataf = '../supertag_data/train.dat'
	params.dev = '../supertag_data/dev.dat'
	params.test = '../supertag_data/test.dat'
	params.batchsize = 10
	params.hidden = hidden
	params.embedsize = 100
	params.eta = eta
	params.L2 = l2
	params.dropout = 1
	params.emb =1	
	params.inf = inf
	

        params.char_embedd_dim = 30
        params.num_filters = num_filters

	params.en_hidden_size = hidden
	params.de_hidden_size= hidden
	params.lstm_layers_num =1
	

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
        else:
                tagger = getTagger('../supertag_data/tagger_400')
        params.num_labels = len(tagger)

        char_dic = getTagger('../supertag_data/char_dic')

        params.char_dic = char_dic

        scale = np.sqrt(3.0 / params.char_embedd_dim)
        char_embedd_table = np.random.uniform(-scale, scale, [len(char_dic), params.char_embedd_dim]).astype(theano.config.floatX)

	
	params.outfile = params.outfile+".num_filters_"+str(num_filters)+'_LearningRate_'+str(params.eta)+ '_inf_' +str(inf) +'_hidden_'+ str(params.hidden)+ '_' + str(tagversion)+ '_'+ str(l2)
                                #examples are shuffled data
	
        train = getSupertagData_and_Char(params.dataf, words, tagger, char_dic)

        dev = getSupertagData_and_Char(params.dev, words, tagger, char_dic, train=False)

        test = getSupertagData_and_Char(params.test, words, tagger, char_dic, train=False)

	
	if (inf ==0) or (inf==1):
                from base_model_selection_ccctag import base_model
		tm = base_model(We, char_embedd_table, params)
		tm.train(train, dev, test, params)

	elif(inf ==2):
                from seq2seq_att_pos import Seq2Seq
                #from seq2seq_att_pos_new import Seq2Seq
                tm = Seq2Seq(We, char_embedd_table, params)
                tm.train(train, dev, test, params)
	elif(inf ==3):
                ##from seq2seq_att_pos_beamsearch import Seq2Seq
                from seq2seq_att_pos_new_beamsearch import Seq2Seq
                tm = Seq2Seq(We, params)
                tm.train(traindata, devdata, testdata, params)


if __name__ == "__main__":
       Base(float(sys.argv[1]), float(sys.argv[2]) , int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
