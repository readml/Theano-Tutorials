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

    """
    FIXME:  Must use a Elementwise Theano Multiplication
    Hidden Outputs should be the individual hidden unit * weight.
    This will allow us to see the functions of the hidden units.

    hidden_outputs = []
    for h_val, weight in zip(h, w_o):
        hidden_outputs.append(T.prod(h_val, weight))

    # output layer with b`asic linear regression
    out = T.sum(hidden_outputs)

    return out, hidden_output
    """
    out = T.dot(h, w_o)
    return out

def create(func, xlow=-1.5, xhigh=1.5, bias=True):
    trX = np.linspace(xlow, xhigh, N_EXAMPLES + 1)

    trY = func(trX) + np.random.randn()*0.05
    
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

out = model(X, w_h, w_o)

cost = T.mean((out - y)**2)
params = [w_h, w_o]
updates = sgd(cost, params)

train = theano.function(inputs=[X, y], outputs=[cost, w_h], updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=out, allow_input_downcast=True)


for i in range(3 * 10**3):
    for X, y in zip(trX, trY):
        # X is a row in trX. X is shape (2,)
        # y is an element of trY. y is shape () 
        cost, w_h = train(X, y)

# def finaloutput():
#     for x in trX
#     retrn y_predict, h_unit0, h_unit1, h_unit2 

# y_predict, h_unit0, h_unit1, h_unit2 = finaloutput()

y_pred = [predict(x) for x in trX]
plt.hold('on')
plt.plot(y_pred, label='predict')
plt.plot(trY, 'r.', label='ground truth')
plt.legend()
plt.show()


