import sys
import warnings
from utils import getGloveWordmap
from params import params
from utils import getSupertagData_and_Char
from utils import getTagger
import random
import numpy as np
import theano


random.seed(1)
np.random.seed(1)

warnings.filterwarnings("ignore")
params = params()

def Base(eta, l3, epoches, warmstart):
	params.outfile = 'CRF_Inf_ccctag'
	params.dataf = '../supertag_data/train.dat'
	params.dev = '../supertag_data/dev.dat'
	params.test = '../supertag_data/test.dat'


        tagversion = 2
        batchsize = 10


        params.epoches = epoches
	params.batchsize = batchsize
        params.hidden = 400
        params.embedsize = 100
        params.emb = 1
        params.eta = eta
        params.dropout = 1


        num_filters = 30
        params.char_embedd_dim = 30
        params.num_filters = num_filters

     
        params.regutype = 0
        params.annealing = 0
        params.L3 = l3
	
        params.hidden_inf = 400

        params.WarmStart = warmstart


	(words, We) = getGloveWordmap('../embedding/glove.6B.100d.txt')
	words.update({'UUUNKKK':0})
	a=[0]*len(We[0])
	newWe = []
	newWe.append(a)
	We = newWe + We
	We = np.asarray(We).astype('float32')
	#print We.shape

	if (tagversion==0):
                tagger = getTagger('../supertag_data/tagger_100')
        elif(tagversion==1):
                tagger = getTagger('../supertag_data/tagger_200')
        else:
                tagger = getTagger('../supertag_data/tagger_400')
        params.num_labels = len(tagger)	

	params.words = words
	params.tagger = tagger
       
        char_dic = getTagger('../supertag_data/char_dic')

        params.char_dic = char_dic

        scale = np.sqrt(3.0 / params.char_embedd_dim)
        char_embedd_table = np.random.uniform(-scale, scale, [len(char_dic), params.char_embedd_dim]).astype(theano.config.floatX)

	params.outfile = params.outfile + '_LearningRate_'+str(params.eta)+ '_'  + str(l3) +'_num_filters_'+ str(num_filters)

        print params.outfile	
        train = getSupertagData_and_Char(params.dataf, words, tagger, char_dic)

        dev = getSupertagData_and_Char(params.dev, words, tagger, char_dic, train=False)

        test = getSupertagData_and_Char(params.test, words, tagger, char_dic, train=False)	

        print len(test[0])

        from model_selection_ccctag_sgd_inference import CRF_model
        tm = CRF_model(We, char_embedd_table, params)
        tm.train(train, dev, test, params)


if __name__ == "__main__":
	Base(float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
