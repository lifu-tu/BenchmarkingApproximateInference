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


def Base(eta, epoches):
	params.outfile = 'Pos_sgd_Inf_'
	params.dataf = '../pos_data/oct27.traindev.proc.cnn'
	params.dev = '../pos_data/oct27.test.proc.cnn'
	params.test = '../pos_data/daily547.proc.cnn'


        l3 = 0
        emb = 0
        params.char_embedd_dim = 30
      
	params.batchsize = 10
        params.hidden = 100
        params.embedsize = 100
        params.emb = emb
        params.eta = eta
        params.dropout = 0
        

        params.regutype = 0
        params.annealing = 0
        params.L3 = l3
	
        params.epoches = epoches

	(words, We) = getWordmap('../embedding/wordvects.tw100w5-m40-it2')
	We = np.asarray(We).astype('float32')
	#print We.shape
	tagger = getTagger('../pos_data/tagger')
	#print tagger
	params.words = words
	params.tagger = tagger
   
        params.num_labels = len(tagger) 
   
        char_dic = getTagger('../pos_data/char_dic')

        params.char_dic = char_dic

		
	params.outfile = params.outfile + '_LearningRate_'+str(params.eta)+ '_epoches_'+ str(epoches)
        print params.outfile

        train = getData_and_Char(params.dataf, words, tagger, char_dic)
        dev = getData_and_Char(params.dev, words, tagger, char_dic)
        test = getData_and_Char(params.test, words, tagger, char_dic)
	

	
        from model_selection_sgd_simple_inference import CRF_model
        tm = CRF_model(We, params)
        tm.train(train, dev, test, params)	


if __name__ == "__main__":
	Base(float(sys.argv[1]), int(sys.argv[2]))
