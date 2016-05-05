
# coding: utf-8

# In[7]:

'''
http://deeplearning.net/tutorial/dA.html
http://deeplearning.net/tutorial/code/dA.py
'''
from __future__ import print_function

import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import load_data

from utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image


# In[8]:

class dA:
    """Denoising Auto-Encoder class (dA)

    A denoising autoencoders tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space. Please refer to Vincent et al.,2008
    for more details. If x is the input then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2)
    computes the projection of the input into the latent space. Equation (3)
    computes the reconstruction of the input, while equation (4) computes the
    reconstruction error.

    .. math::

        ~x ~ q_D(~x}|x)                                     (1)

        y = s(W * ~x + b)                                           (2)

        z = s(W' y  + b')                                                (3)

        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)

    """
    def __init__(self,
                numpy_rng,
                theano_rng=None,
                input=None,
                n_visible=784,
                n_hidden=500,
                W=None,
                bhid=None,
                bvis=None):
        self.n_visible=n_visible
        self.n_hidden=n_hidden
        
        if not theano_rng:
            theano_rng=RandomStream(numpy_rng.randint(2 ** 30))
            
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W=numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)),
                dtype=theano.config.floatX)
            W=theano.shared(value=initial_W, name='W', borrow=True)
            
        if not bvis:
            bvis=theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True)
            
        if not bhid:
            bhid=theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True)
            
        self.W=W
        # b corresponds to the bias of the hidden
        self.b=bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime=bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime=self.W.T
        self.theano_rng=theano_rng
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input
        
        self.params=[self.W, self.b, self.b_prime]
    
    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input,self.W)+self.b)
    
    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden,self.W_prime)+self.b_prime)
    
    def get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.
        """
        return self.theano_rng.binomial(size=input.shape,n=1,
                                       p=1-corruption_level,
                                       dtype=theano.config.floatX)*input
    
    def get_cost_updates(self,corruption_level,learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """
        
        tilde_x=self.get_corrupted_input(self.x, corruption_level)
        y=self.get_hidden_values(tilde_x)
        z=self.get_reconstructed_input(y)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        L=-T.sum(self.x*T.log(z)+(1-self.x)*T.log(1-z),axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost=T.mean(L)
        
        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]
        return(cost,updates)


# In[25]:

def build_mdl(n_train_batches,batch_size,
             corruption_level,
             learning_rate,train_set_x):
    print('... building the model')
    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    
    rng=numpy.random.RandomState(123)
    theano_rng=RandomStreams(rng.randint(2 ** 30))
    
    da=dA(numpy_rng=rng,
         theano_rng=theano_rng,
         input=x,
         n_visible=28*28,
         n_hidden=500)
    
    cost,updates=da.get_cost_updates(
        corruption_level=corruption_level,
        learning_rate=learning_rate)
    
    # about givens see: theano.pdf page 40 (actual 44 / 495)
    train_da=theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x:train_set_x[index*batch_size:(index+1)*batch_size]
        }
    )
    
    return train_da,da

def train_mdl(train_da,da,training_epochs,
              n_train_batches,corruption_level,
              output_folder):
    
    start_time=timeit.default_timer()
    for epoch in range(training_epochs):
        # go through training set
        c=[]
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))
            if batch_index%500==0:
                sys.stdout.write('%d '%batch_index)
        print('\nTraining epoch %d(%d batches), cost %f'% (epoch, n_train_batches, numpy.mean(c)))
    
    end_time=timeit.default_timer()
    training_time = (end_time - start_time)
    print(('The code for file ' +
           os.path.split(os.path.realpath('__file__'))[1] +
           ' ran for %.2fm' % ((training_time) / 60.)), file=sys.stderr)
    image = Image.fromarray(
        tile_raster_images(X=da.W.get_value(borrow=True).T,
                           img_shape=(28, 28), tile_shape=(10, 10),
                           tile_spacing=(1, 1)))
    image.save(output_folder+'/filters_corruption_%0.2f.png'%corruption_level)


# In[ ]:

def test_dA(learning_rate=0.1,training_epochs=15,
           dataset='mnist.pkl.gz',
           batch_size=20,
           output_folder='dA_plots'):
    """
    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing
                          AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

    :type dataset: string
    :param dataset: path to the picked dataset
    """
    datasets=load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
        
    #os.chdir(output_folder)
    #os.chdir('../')

    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################    
    train_da,da=build_mdl(n_train_batches,batch_size,
                       0.,
                       learning_rate,train_set_x)

    train_mdl(train_da,da,training_epochs, 
              n_train_batches, 0.0,
              output_folder)
    #####################################
    # BUILDING THE MODEL CORRUPTION 30% #
    #####################################
    train_da,da=build_mdl(n_train_batches,batch_size,
                       0.3,
                       learning_rate,train_set_x)
    
    train_mdl(train_da,da,training_epochs, 
              n_train_batches, 0.3,
              output_folder)
    
if __name__=='__main__':
    test_dA()


# In[ ]:



