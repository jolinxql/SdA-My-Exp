---
title: Classifying MNIST Digits Using Stacked Denoising AutoEncoders - My Experiments
date: 2016-05-04 01:29:38
tags:
    - Theano
    - DeepLearning
---
by Qiaolin Xia

>**Note**

>I wrote this article based on my own experiments when I was practicing using Theano. I implemented two classifiers for this article -- *Simple MLP* and *MLP+SdA (MLP finetuning after SdA pretraining)*. Experimental results and my analysis has also been given. If you intend to run the code, my code is available for download on my Github. The code is slightly different with what deeplearning.net provided. I've only tested it with ipython notebook (python2).
>
>Especially if you do not have experience with autoencoders, I recommend reading this tutorial [Denoising Autoencoders](http://deeplearning.net/tutorial/dA.html#da) before going any further.


## Content

- **Introduction**
- **Stacked Auto-Encoders**
  - Autoencoders
  - Denoising Auto-Encoders
  - Stacked Auto-Encoders
- **Finetuning using MLP**
- **Criteria for Evaluating Performance**
- **Configuration of two models**
  - Dataset Size
  - Baseline - MLP
  - Extension - MLP finetuning with SdA pretraining (SdA+MLP)
- **Experimental Results**
- **Conclusion**
- **References**



## **Introduction**

**The Stacked Denoising Autoencoder (SdA)** is an extension of the stacked autoencoder [[Bengio01](http://deeplearning.net/tutorial/references.html#bengio07)]and it was introduced in [[Vincent08](http://deeplearning.net/tutorial/references.html#vincent08)].

In this article, we show how **the SdA and the Multilayer Perceptron (MLP)** can be jointly used to implement a MNIST digits classifier.

**The MNIST dataset** consists of handwritten digit images and it is divided in 60,000 examples for the training set and 10,000 examples for testing. In many papers as well as in this tutorial, the official training set of 60,000 is divided into an actual training set of 50,000 examples and 10,000 validation examples (for selecting hyper-parameters like learning rate and size of the model). All digit images have been size-normalized and centered in a fixed size image of 28 x 28 pixels. In the original dataset each pixel of the image is represented by a value between 0 and 255, where 0 is black, 255 is white and anything in between is a different shade of grey.

For convenience **we use pickled dataset [mnist.pkl.gz](http://deeplearning.net/data/mnist/mnist.pkl.gz)** to make it easier to use in python. 

## **Stacked Auto-Encoders**
  
### Autoencoders

An autoencoder takes an input $x\in{(0,1)}^{d'}$ and first maps it (with an *encoder*) to a hidden representation  through a deterministic mapping, e.g.: 

$$y=s(Wx+b)$$

Where s is a non-linearity such as the sigmoid. The latent representation ${y}$, or code is then mapped back (with a decoder) into a reconstruction ${z}$ of the same shape as ${x}$. The mapping happens through a similar transformation, e.g.:

$${z} = s({W'}{y} + {b'})$$

The traditional squared error $L({x}{z}) = || {x}-{z} ||^2$, can be used. If the input is interpreted as either bit vectors or vectors of bit probabilities, cross-entropy of the reconstruction can be used:

$$L_{H} ({x}, {z}) = - \sum^d_{k=1}[{x}_k \log{z}_k + (1 -{x}_k)\log(1 -{z}_k)]$$

### Denoising Auto-Encoders

The idea behind denoising autoencoders is simple. In order to force the hidden layer to discover more robust features and prevent it from simply learning the identity, we train the autoencoder to reconstruct the input from a corrupted version of it.

To convert the autoencoder class into a denoising autoencoder class, all we need to do is to add a stochastic corruption step operating on the input. 

### Stacked Auto-Encoders

Denoising autoencoders can be stacked to form a deep network by feeding the latent representation (output code) of the denoising autoencoder found on the layer below as input to the current layer. The unsupervised pre-training of such an architecture is done one layer at a time. Each layer is trained as a denoising autoencoder by minimizing the error in reconstructing its input (which is the output code of the previous layer). Once the first k layers are trained, we can train the k+1-th layer because we can now compute the code or latent representation from the layer below.

## **Finetuning using MLP**

Once all layers are pre-trained, the network goes through a second stage of training called fine-tuning. Here we consider supervised fine-tuning where we want to minimize prediction error on a supervised task. For this, we first add a logistic regression layer on top of the network (more precisely on the output code of the output layer). We then train the entire network as we would train a multilayer perceptron. At this point, we only consider the encoding parts of each auto-encoder. This stage is supervised, since now we use the target class during training. (See the [Multilayer Perceptron](http://deeplearning.net/tutorial/mlp.html#mlp) for details on the multilayer perceptron.)

## **Criteria for Evaluating Performance**

Learning optimal model parameters involves minimizing a loss function. In the case of multi-class logistic regression, it is very common to use the negative log-likelihood as the loss. This is equivalent to maximizing the likelihood of the data set $\cal{D}$ under the model parameterized by $\theta$. Let us first start by defining the likelihood $\cal{L}$ and loss $\ell$:

$$\cal{L} (\theta=\{W,b\}, \mathcal{D}) =
  \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
\ell (\theta=\{W,b\}, \mathcal{D}) = - \mathcal{L} (\theta=\{W,b\}, \mathcal{D})$$

This article we use the method of stochastic gradient method with mini-batches (MSGD).

## **Configuration of two models**

### Dataset Size

|dataset| size |
|:---:|:--:|
|train  |50000 |
|dev    |10000 |
|test   |10000 |

### Baseline - MLP

|  |learning rate|max epochs|batch_size|hidden layer size|L1_reg|L2_reg|
|--|:-----------:|:--------:|:--------:|:---------------:|:----:|:----:|
|Simple MLP|0.01|1000|20|500|0.01|0.0001|

### Extension - MLP finetuning with SdA pretraining (SdA+MLP)

| |learning rate|max epochs|batch_size|hidden layer size|corruption_level|
|---------|:---:|:--:|:--:|:-------:|:--:|
|Pretrain |0.001| 10 | 50 | 500 500 | 0.2|
|Finetune |0.1  |1000| 20 | 500 500 | -  |

Early-Stopping heuristics : geometrically increasing amount of patience


## **Experimental Results and Analysis**

These are the results of the two models.

||Simple MLP |SdA+MLP|
|:--|:--:|:--:|
|**best performance after 1st finetune epoch**|dev error 9.62% with test error 10.09%|dev error 8.19% with test error 9.07%|
|**final result**|dev error 1.68% with test error 1.65%|dev error 1.72% with test error 1.69%|
|**total running time**|247.13m|535.14m|
|**pretrain running time**|-|20.05m|
|**finetune running time**|-|515.09m|
|**early-stop at **|1000 epochs|663 epochs|

The results show that final performance of *SdA+MLP* on the test set is slightly poor, but it does not necessarily mean that *Simple MLP* is better than the model pretrained by an autoencoder, because the total training epochs (see the last line in the table) of *Simple MLP* is about twice as the epochs of *SdA+MLP*. What's more, without pretraining, the best performance of *Simple MLP* after the 1st finetuning epoch is obviously poor.


## **Conclusion**

In this article, I implemented two classifiers for MNIST digits dataset. The baseline model (*Simple MLP*) is based on a 1-layer MLP and the extension (SdA+MLP) is a 2-layer MLP with a stack denoising autoencoder using shared weights for pretraining. The experiment shows that with pretraining, the model's best performance of after the 1st finetuning epoch is better. However, because we didn't make great efforts to adjust hyper-parameters, the final result of SdA+MLP is not as good as we expect, though the differenc between to classifier is very small.

## **References**

http://deeplearning.net/tutorial/SdA.html
http://deeplearning.net/tutorial/mlp.html
http://deeplearning.net/tutorial/gettingstarted.html#l1-l2-regularization 