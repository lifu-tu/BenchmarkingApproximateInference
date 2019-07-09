import sys
import warnings
from utils import getWordmap
from params import params
from utils import getData
#from utils import getUnlabeledData
from utils import getTagger
import random
import numpy as np
from model_selection_inference import CRF_model
#from model_selection_inference_seq2seq import CRF_seq2seq_model

random.seed(1)
np.random.seed(1)

warnings.filterwarnings("ignore")
params = params()

#def Base(eta, l3, emb, batchsize, inf, regutype, annealing):
def Base(eta, l3, emb, batchsize, inf, hidden_inf):
	params.outfile = 'h_pos_CRF_Inf_'
	params.dataf = '../pos_data/oct27.traindev.proc.cnn'
	params.dev = '../pos_data/oct27.test.proc.cnn'
	params.test = '../pos_data/daily547.proc.cnn'

	params.batchsize = batchsize
        params.hidden = 100
        params.embedsize = 100
        params.emb = emb
        params.eta = eta
        params.dropout = 0
	params.hidden_inf = hidden_inf
	params.small = 0


        params.inf = inf
        params.regutype = 0
        params.annealing = 0
        params.L3 = l3
	

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
	params.words = words
	params.tagger = tagger
   
        params.num_labels = len(tagger) 
   
	#params.outfile = params.outfile+".Batchsize"+'_'+str(params.batchsize)+'_dropout_'+ str(params.dropout) + '_LearningRate_'+str(params.eta)+ '_'  + str(l3) +'_emb_'+ str(emb)+ '_inf_'+ str(params.inf)+ '_regutype_'+ str(params.regutype)+ '_annealing_'+ str(params.annealing)
	
	params.outfile = params.outfile+".Batchsize"+'_'+str(params.batchsize)+'_dropout_'+ str(params.dropout) + '_LearningRate_'+str(params.eta)+ '_'  + str(l3) +'_emb_'+ str(emb)+ '_inf_'+ str(params.inf)+ '_hidden_inf_'+ str(params.hidden_inf)

	traindata = getData(params.dataf, words, tagger)
	trainx0, trainy0 = traindata
 	devdata = getData(params.dev, words, tagger)
	devx0, devy0 = devdata
	print 'dev set',  len(devx0)
	testdata = getData(params.test, words, tagger)
	testx0, testy0 = testdata	

	print 'test set', len(testx0)
	#print Y
	print "Using Training Data"+params.dataf
	print "Using Word Embeddings with Dimension "+str(params.embedsize)
	print "Saving models to: "+params.outfile
	#lm = LM_model(params)
	#lm.train(trainy0, devy0, params)	

	if (inf ==0) or (inf==1):
		tm = CRF_model(We, params)
		tm.train(trainx0, trainy0, devx0, devy0, testx0, testy0, params)
	elif(inf==2):
		from model_selection_inference_seq2seq_h import CRF_seq2seq_model

		params.de_hidden_size = hidden_inf
		#params.de_hidden_size = 50 
		params.outfile = 'h_pos_de_hidden_size_' + str(params.de_hidden_size)+ '_'+ params.outfile
		tm = CRF_seq2seq_model(We, params)
                tm.train(trainx0, trainy0, devx0, devy0, testx0, testy0, params)
	elif(inf==3):
                from model_selection_inference_seq2seq_h_beamsearch import CRF_seq2seq_model

                params.de_hidden_size = hidden_inf
		#params.de_hidden_size = 50
                params.outfile = 'h_pos_de_hidden_size_' + str(params.de_hidden_size)+ '_'+ params.outfile
                tm = CRF_seq2seq_model(We, params)
                tm.train(trainx0, trainy0, devx0, devy0, testx0, testy0, params)



if __name__ == "__main__":
	Base(float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
