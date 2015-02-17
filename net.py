import theano
from theano import tensor as T
import numpy as np
import matplotlib.pyplot as plt

N_EXAMPLES = 50
EPOCHS = 2 * 10**2
LEARNING_RATE = 0.1
WEIGHT_INIT_STD = 0.5
X_DOMAIN = [-5.0, 5.0]


trX, trY = sinusoid()

print trX.shape
print trY.shape

def x2():
    return create(func=np.square)

def absolute():
    return create(func=np.abs)

def sinusoid():
    return create(func=np.sin)

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape, std=WEIGHT_INIT_STD):
    return theano.shared(floatX(np.random.randn(*shape) * std))

# I don't currently understand if this is doing batches, or single training points
def sgd(cost, params, lr=LEARNING_RATE):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates

def model(X, w_h, w_o):
    """ do a dot product of weights and final hidden nodes to get the final
    output
    """
    # hidden layer with sigmoid activation
    h = T.nnet.sigmoid(T.dot(X, w_h))

    out = T.dot(h, w_o)
    return out

def hidden_out(X, w_h, w_o):
    """ do an element wise multiply of final hidden node * weight.  This will
    give the final output values for each node at the final layer.  These values,
    added, would give the final output
    """
    h = T.nnet.sigmoid(T.dot(X, w_h))

    # transpose to do element wise multiply
    return h * w_o.T

def create(func, xlow=X_DOMAIN[0], xhigh=X_DOMAIN[1], bias=True, std=0.5):
    trX = np.linspace(xlow, xhigh, N_EXAMPLES + 1)

    trY = func(trX) + np.random.randn()*std
    
    if bias:
        trX = np.column_stack((np.ones(trX.shape[0]), trX))

    return trX, trY

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
w_h = init_weights((2, 4))
w_o = init_weights((4, 1))

out= model(X, w_h, w_o)
hidden_out = hidden_out(X, w_h, w_o)

cost = T.mean((out - y)**2)
params = [w_h, w_o]
updates = sgd(cost, params)

train = theano.function(inputs=[X, y], outputs=[cost, w_h], updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=out, allow_input_downcast=True)
predict_hidden_out = theano.function(inputs=[X], outputs=hidden_out, allow_input_downcast=True)

for i in range(EPOCHS):
    for X, y in zip(trX, trY):
        # X is a row in trX. X is shape (2,)
        # y is an element of trY. y is shape () 
        cost, w_h = train(X, y)

y_pred = [predict(x) for x in trX]

# this captures the hidden output units in a matrix
# where a row is the hidden unit outputs for a data example
    # and a column is the value(s) for a particular hidden unit
hidden_outputs = np.array([predict_hidden_out(x)[0] for x in trX])

plt.hold('on')
plt.plot(trX[:,1], y_pred, label='predict')
plt.plot(trX[:,1], trY, 'r.', label='ground truth')

for node_j in range(hidden_outputs.shape[1]):
    plt.plot(trX[:,1], hidden_outputs[:,node_j], label='hidden unit %d' % node_j)

plt.legend()
plt.figtext(.5,.95,'learning rate | weight init std | num examples | epochs', fontsize=14, ha='center')
plt.figtext(.5,.91,'{} | {} | {}'.format(LEARNING_RATE, WEIGHT_INIT_STD, N_EXAMPLES, EPOCHS),fontsize=14,ha='center')
plt.show()


