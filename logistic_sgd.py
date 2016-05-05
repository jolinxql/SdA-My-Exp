
# coding: utf-8

# In[2]:

'''
http://deeplearning.net/tutorial/logreg.html#logreg
http://deeplearning.net/tutorial/code/logistic_sgd.py
'''
from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T


# In[3]:

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """
    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie. e.g. for MNIST, is 28*28=784

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        self.W=theano.shared(
            value=numpy.zeros( (n_in,n_out), dtype=theano.config.floatX),
            name='W',
            borrow=True)
        self.b=theano.shared(
            value=numpy.zeros( (n_out,), dtype=theano.config.floatX),
            name='b',
            borrow=True)
        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x=T.nnet.softmax(T.dot(input,self.W)+self.b)
        
        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input
    
    def negative_log_likelihood(self, y):
        # p_y_given_x=softmax(input(dot)W)+b)
        sigma = T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]
        '''Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size'''
        return -T.mean(sigma)
    
    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        
        # check if y has same dimension of y_pred
        # y TensorType(int32, vector)
        # y_pred TensorType(int64, vector)
        # y.ndim=1
        if y.ndim!=self.y_pred.ndim:
            raise TypeError('y and self.y_pred should have the same shape',
                           ('y',y.type,'y_pred',self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def load_data(dataset):
    
    def shared_dataset(data_xy, borrow=True):
        '''
        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        '''
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        ret_x=shared_x # theano.tensor.sharedvar.TensorSharedVariable
        ret_y=T.cast(shared_y, 'int32') # theano.tensor.var.TensorVariable
        '''Note: T.cast is like a layer.'''
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return ret_x, ret_y
    
    with gzip.open(dataset, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f)

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    return rval


# In[7]:

def build_mdl(train_set_x, train_set_y,
                valid_set_x, valid_set_y,
                test_set_x, test_set_y,
                learning_rate,
                batch_size):

    print('... building the model')
    # allocate symbolic variables for the data
    index=T.lscalar() # index to a [mini]batch
    
    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x=T.matrix('x')
    y=T.ivector('y') # labels, presented as 1 dimentsion int vector
    
    # Each MNIST image has size 28*28
    classifier=LogisticRegression(input=x, n_in=28*28, n_out=10)
    
    cost = classifier.negative_log_likelihood(y)
    
    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model=theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        })
    
    validate_model=theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        })
    
    # compute the gradient of cost with respect to theta = (W,b)
    # grad_W= d(cost)/d(W)
    g_W=T.grad(cost=cost,wrt=classifier.W)
    g_b=T.grad(cost=cost,wrt=classifier.b)
    
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates=[(classifier.W, classifier.W - learning_rate * g_W),
            (classifier.b, classifier.b - learning_rate * g_b)]
    
    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model=theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        })
    
    return train_model,validate_model,test_model,classifier

def train_mdl(n_train_batches,n_valid_batches,n_test_batches,classifier,
              n_epochs,
              train_model,validate_model,test_model):
    print('... training the model')
    # for early stop
    patience=5000  # look as this many examples regardless
    patience_increase=2 # wait this much longer when a new best is found
    improvement_threshold=0.995 # a relative improvement of this much is
                                  # considered significant
    validation_frequency=min(n_train_batches, patience // 2) # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
    print('patience:%d, patience_inc:%d'%(patience,patience_increase))
    print('improve_thresh:%0.2f, valid_freq:%d'%(improvement_threshold,validation_frequency))
    
    best_validation_error = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()
    
    done_looping = False
    epoch = 0
    
    
    while (epoch < n_epochs) and (not done_looping):
        
        epoch = epoch + 1
        
        for minibatch_index in range(n_train_batches):
            
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
            
            minibatch_avg_cost = train_model(minibatch_index)
            
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_errors = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_error = numpy.mean(validation_errors)
                
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                    (epoch,minibatch_index + 1,n_train_batches,
                     this_validation_error * 100.))
                
                # if we got the best validation score until now
                if this_validation_error < best_validation_error:
                    if this_validation_error < best_validation_error * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                        print('    **significant improvement at iter:%d patience set to:%d'                              %(iter + 1,patience))
                        
                    best_validation_error = this_validation_error   
                    test_errors = [test_model(i) for i in range(n_test_batches)]
                    test_score = numpy.mean(test_errors)
                    
                    print(('    test error of best model on validation %f %%')%(test_score * 100.))
                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(classifier, f)
                        
            if patience <= iter:
                done_looping = True
                break
    end_time = timeit.default_timer()
    print(('Optimization complete with best validation error score of %f %%,'
        'with test performance %f %%')% (best_validation_error * 100., test_score * 100.))
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
        os.path.split(os.path.realpath('__file__'))[1] +
        ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)

def sgd_optim_mnist(learning_rate=0.13,
                   n_epochs=1000,
                   dataset='mnist.pkl.gz',
                   batch_size=600):
    datasets=load_data(dataset)
    
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    # (50000L, 784L) (50000L,)
    # (10000L, 784L) (10000L,)
    # (10000L, 784L) (10000L,)
    
    train_model,validate_model,test_model,classifier=build_mdl(train_set_x, train_set_y,
                valid_set_x, valid_set_y,
                test_set_x, test_set_y,
                learning_rate,
                batch_size)
    
    ###############
    # TRAIN MODEL #
    ###############
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
    
    train_mdl(n_train_batches,n_valid_batches,n_test_batches,classifier,
              n_epochs,
              train_model,validate_model,test_model)


# In[8]:

def predict():
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    # load the saved model
    classifier = pickle.load(open('best_model.pkl'))

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)

    # We can test it on some examples from test test
    dataset='mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[:10])
    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values) # TensorSharedVariable
    print(test_set_y[:10].eval()) # we use eval() to show TensorVariable


# In[10]:

if __name__=='__main__':
    sgd_optim_mnist()


# In[11]:

if __name__=='__main__':
    predict()


# In[ ]:



