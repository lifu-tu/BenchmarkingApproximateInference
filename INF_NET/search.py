import numpy as np
import copy

def gen_beam_sample( f_next, ctx, k, de_hidden_size, voc_size):

    # k is the beam size we have
    sample = []
    sample_score = []

    maxlen = ctx.shape[0]

    hyp_samples = [[]] * 1
    hyp_scores = np.zeros(1).astype('float32')
    #hyp_states = []
    

    next_state = np.zeros((1, voc_size+1),  dtype='float32')   
    next_state[0, -1] = 1.    

    next_h = np.zeros((1, 1, de_hidden_size)).astype('float32')
    next_c = np.zeros((1, 1, de_hidden_size)).astype('float32')
	
    for ii in xrange(maxlen):

	batch_size = next_state.shape[0]

        next_ctx = np.tile(ctx[ii,:], [batch_size, 1])
	
	#print next_ctx.shape, next_state.shape, next_h.shape, next_c.shape
        inps = [next_ctx, next_state, next_h, next_c]
        ret = f_next(*inps)
        next_state, next_h, next_c = ret[0], ret[1], ret[2]
	#print next_state
          
        cand_scores = hyp_scores[:, None] - np.log(next_state[:,:-1])
        cand_flat = cand_scores.flatten()
        ranks_flat = cand_flat.argsort()[:k]

    
        trans_indices = ranks_flat / voc_size
        word_indices = ranks_flat % voc_size
        costs = cand_flat[ranks_flat]

        new_hyp_samples = []
        new_hyp_scores = np.zeros(k).astype('float32')
        new_hyp_states = []
	new_hyp_h = []
	new_hyp_c = []

        for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):

                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])

                new_hyp_states.append(copy.copy(next_state[ti]))
		new_hyp_h.append(copy.copy(next_h[0, ti]))
		new_hyp_c.append(copy.copy(next_c[0, ti]))

	        
	hyp_samples = [a for a in new_hyp_samples]
        hyp_scores = np.array(new_hyp_scores)

        next_state = np.array(new_hyp_states)

	new_h = np.array(new_hyp_h)
	next_h = new_h.reshape((1, k, de_hidden_size))
	new_c = np.array(new_hyp_c)
	next_c = new_c.reshape((1, k, de_hidden_size))

	
    
    return hyp_samples, hyp_scores
