import theano
from theano import tensor as T
import numpy as np
import matplotlib.pyplot as plt

N_EXAMPLES = 50

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

# I don't currently understand if this is doing batches, or single training points
def sgd(cost, params, lr=0.03):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates

def model(X, w_h, w_o):
    # hidden layer with sigmoid activation
    h = T.nnet.sigmoid(T.dot(X, w_h))

    out = T.dot(h, w_o)
    return out

def hidden_out(X, w_h, w_o):
    h = T.nnet.sigmoid(T.dot(X, w_h))

    # transpose to do element wise mult
    return h * w_o.T

def create(func, xlow=-5.0, xhigh=5.0, bias=True):
    trX = np.linspace(xlow, xhigh, N_EXAMPLES + 1)

    trY = func(trX) + np.random.randn()*0.1
    
    if bias:
        trX = np.column_stack((np.ones(trX.shape[0]), trX))

    return trX, trY

def x2():
    return create(func=np.square)

def absolute():
    return create(func=np.abs)

def sinusoid():
    return create(func=np.sin)

trX, trY = sinusoid()

print trX.shape
print trY.shape

# will train one row of X
X = T.fvector(name='X') 

# will train one single value of y
y = T.scalar(name='y')

"""
output nodes   o
              /|\
hidden nodes o o o 
              \/\/
input nodes   o  o 
"""
w_h = init_weights((2, 3))
w_o = init_weights((3, 1))

out= model(X, w_h, w_o)
hidden_out = hidden_out(X, w_h, w_o)

cost = T.mean((out - y)**2)
params = [w_h, w_o]
updates = sgd(cost, params)

train = theano.function(inputs=[X, y], outputs=[cost, w_h], updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=out, allow_input_downcast=True)
predict_hidden_out = theano.function(inputs=[X], outputs=hidden_out, allow_input_downcast=True)

for i in range(5 * 10**3):
    for X, y in zip(trX, trY):
        # X is a row in trX. X is shape (2,)
        # y is an element of trY. y is shape () 
        cost, w_h = train(X, y)

y_pred = [predict(x) for x in trX]

hidden_outputs = np.array([predict_hidden_out(x)[0] for x in trX])

plt.hold('on')
plt.plot(trX[:,1], y_pred, label='predict')
plt.plot(trX[:,1], trY, 'r.', label='ground truth')

for node_j in range(hidden_outputs.shape[1]):
    plt.plot(trX[:,1], hidden_outputs[:,node_j], label='hidden unit %d' % node_j)

plt.legend()
plt.show()


