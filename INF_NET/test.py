import numpy as np
from theano import tensor as T
import theano

"""
a = T.ftensor3()
b = T.ivector()

c = a[T.arange(b.shape[0]),b]

test = theano.function([a,b], c)

a0 = np.random.uniform(-0.01, 0.01, (3, 4, 5)).astype('float32')
b0 = np.random.randint(4, size=(3)).astype('int32')
c0 = test(a0, b0)
"""
a = T.fvector()
#a = T.fmatrix()
#b = T.ftensor3()

c = T.nonzero(a, True)
d = T.nonzero_values(a)

test = theano.function([a], [c,d])

#a0 = np.random.uniform(-0.01, 0.01, (2, 3)).astype('float32')
#b0 = np.random.uniform(-0.01, 0.01, (2, 3, 3)).astype('float32')

#b0 = np.random.randint(0, 4, size=(4)).astype('int32')
#c0 = np.random.randint(0, 4, size=(4)).astype('int32')
a0 = np.array([1,2,0,0]).astype('float32')

c0,d0 = test(a0)



#print a0
#print b0
print c0, d0
