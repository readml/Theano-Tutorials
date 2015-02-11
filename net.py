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

    # output layer with basic linear regression
    out = T.dot(h, w_o)
    return out

# trX, teX, trY, teY = mnist(onehot=True)
def x2(xlow=-1.0, xhigh=1.0, bias=True):
    trX = np.linspace(xlow, xhigh, N_EXAMPLES + 1)
    # teX = np.linspace(xlow + .05, xhigh + .05, N_EXAMPLES + 1)

    teY = trX ** 2
    trY = teY + np.random.randn()*0.01
    
    if bias:
        trX = np.column_stack((np.ones(trX.shape[0]), trX))

    return trX, trY

trX, trY = x2()

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

train = theano.function(inputs=[X, y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=out, allow_input_downcast=True)


for i in range(10**4):
    # batch_size = 2
    # for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
    #     cost = train(trX[start:end], trY[start:end])
    for X, y in zip(trX, trY):
        cost = train(X, y)

plt.hold('on')
plt.plot([predict(x) for x in trX], label='predict')
plt.plot(trY, label='ground truth', '.')
plt.show()


