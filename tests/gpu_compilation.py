from __future__ import absolute_import, print_function, division
import os
os.environ['THEANO_FLAGS'] = "device=cuda"

# import numpy as np
# import theano
# import theano.tensor as tt
#
# theano.config.floatX = 'float32'
# print(f'Using {theano.config.device}')
#
# rng = np.random
#
# N = 400
# feats = 784
# D = (rng.randn(N, feats).astype(theano.config.floatX),
#     rng.randint(size=N, low=0, high=2).astype(theano.config.floatX))
# training_steps = 10000
#
# # Declare Theano symbolic variables
# x = theano.shared(D[0], name="x")
# y = theano.shared(D[1], name="y")
# w = theano.shared(rng.randn(feats).astype(theano.config.floatX), name="w")
# b = theano.shared(np.asarray(0., dtype=theano.config.floatX), name="b")
# x.tag.test_value = D[0]
# y.tag.test_value = D[1]
# #print "Initial model:"
# #print w.get_value(), b.get_value()
#
# # Construct Theano expression graph
# p_1 = 1 / (1 + tt.exp(-tt.dot(x, w) - b))  # Probability of having a one
# prediction = p_1 > 0.5  # The prediction that is done: 0 or 1
# xent = -y * tt.log(p_1) - (1 - y) * tt.log(1 - p_1)  # Cross-entropy
# cost = tt.cast(xent.mean(), 'float32') + \
#     0.01 * (w ** 2).sum()  # The cost to optimize
# gw, gb = tt.grad(cost, [w, b])
#
# # Compile expressions to functions
# train = theano.function(
#             inputs=[],
#             outputs=[prediction, xent],
#             updates=[(w, w - 0.01 * gw), (b, b - 0.01 * gb)],
#             name="train")
# predict = theano.function(inputs=[], outputs=prediction,
#             name="predict")
#
# if any([n.op.__class__.__name__ in ['Gemv', 'CGemv', 'Gemm', 'CGemm'] for n in
# train.maker.fgraph.toposort()]):
#     print('Used the cpu')
# elif any([n.op.__class__.__name__ in ['GpuGemm', 'GpuGemv'] for n in
# train.maker.fgraph.toposort()]):
#     print('Used the gpu')
# else:
#     print('ERROR, not able to tell if theano used the cpu or the gpu')
#     print(train.maker.fgraph.toposort())
#
# for i in range(training_steps):
#     pred, err = train()
# #print "Final model:"
# #print w.get_value(), b.get_value()
#
# print("target values for D")
# print(D[1])
#
# print("prediction on D")
# print(predict())
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
fig = plt.figure()
gs = gridspec.GridSpec(2, 2, height_ratios=[2,1], width_ratios=[1,2])
for g in gs:
    print(g)
    ax = fig.add_subplot(g)
fig.tight_layout()