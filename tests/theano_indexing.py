import theano.tensor as T
from theano import function
val = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
v = T.ivector('v')
f = function([v], [T.set_subtensor(v[1:4], [11, 12, 13])]) # Create a function that just returns the input
# Evaluate the function
print(f(val))