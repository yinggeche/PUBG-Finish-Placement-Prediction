import sys
import argparse
import numpy as np
from operator import add
from time import time
from pyspark import SparkContext

parser = argparse.ArgumentParser(description = 'Parallel Ridge Regression.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--traindata',default=None, help='Input file containing (x,y) pairs, used to train a linear model')
parser.add_argument('--testdata',default=None, help='Input file containing (x,y) pairs, used to test a linear model')
parser.add_argument('--beta', default='beta', help='File where beta is stored (when training) and read from (when testing)')
parser.add_argument('--lam', type=float,default=0.0, help='Regularization parameter λ')
parser.add_argument('--max_iter', type=int,default=100, help='Maximum number of iterations')
parser.add_argument('--eps', type=float, default=0.01, help='ε-tolerance. If the l2_norm gradient is smaller than ε, gradient descent terminates.')
parser.add_argument('--N',type=int,default=2,help='Level of parallelism')

verbosity_group = parser.add_mutually_exclusive_group(required=False)
verbosity_group.add_argument('--verbose', dest='verbose', action='store_true')
verbosity_group.add_argument('--silent', dest='verbose', action='store_false')
parser.set_defaults(verbose=True)

args = parser.parse_args()

sc = SparkContext(appName='Parallel Ridge Regression')

if not args.verbose :
    sc.setLogLevel("ERROR")

beta = None

def readBeta(input):
    with open(input,'r') as fh:
        str_list = fh.read()\
                   .strip()\
        	   .split(',')
        return np.array( [float(val) for val in str_list] )

def writeBeta(output,beta):
    with open(output,'w') as fh:
        fh.write(','.join(map(str, beta.tolist()))+'\n')

def estimateGrad(fun,x,delta):
     d = len(x)
     grad = np.zeros(d)
     for i in range(d):
         e = np.zeros(d)
         e[i] = 1.0
         grad[i] = (fun(x+delta*e) - fun(x))/delta
     return grad

def lineSearch(fun,x,grad,a=0.2,b=0.6):
    t = 1.0
    while fun(x-t*grad) > fun(x)- a * t *np.dot(grad,grad):
        t = b * t
    return t

def predict(x,beta):
    return np.dot(x,beta)


def f(x,y,beta):
    return (y-predict(x,beta))**2

def localGradient(x,y,beta):
    return -2*(y-predict(x,beta))*x

def F(data,beta,lam = 0):
    n = data.count()
    MSE=1./n*data.map(lambda (x,y):f(x,y,beta))\
           .reduce(add)

    return MSE + lam * np.dot(beta,beta)

def gradient(data,beta,lam = 0):
    n = data.count()
    grad = data.map(lambda (x,y): localGradient(x,y,beta))\
            .reduce(add)
    grad = 1./n*grad +  2 * lam * beta
    return grad

if (args.traindata):
    # Train a linear model β from data with regularization parameter λ, and store it in beta
    print 'Reading training data from',args.traindata
    data = sc.textFile(args.traindata)\
             .map(lambda line:line.split(','))\
             .map(lambda words:(words[:-1],words[-1]))\
             .map(lambda (features,target): (np.array([ float(x) for x in features]),float(target)))
    data = data.repartition(args.N).cache()

    x,y = data.take(1)[0]
    beta0 = np.zeros(len(x))

    print 'Training on data from',args.traindata,'with λ =',args.lam,', ε =',args.eps,', max iter = ',args.max_iter
    beta_0=beta0
    lam=args.lam
    max_iter=args.max_iter
    eps=args.eps
    k = 0
    gradNorm = 2*eps
    beta = beta_0
    start = time()
    while k<max_iter and gradNorm > eps:
        grad = gradient(data,beta,lam)
        obj = F(data,beta,lam)
        fun = lambda x: F(data,x,lam)
        gamma = lineSearch(fun,beta,grad)
        beta = beta - gamma * grad
        gradNorm = np.linalg.norm(grad)
        print 'k = ',k,'\tt = ',time()-start,'\tF(β_k) = ',obj,'\t||∇F(β_k)||_2=',gradNorm
        k = k + 1
    print 'Algorithm ran for',k,'iterations. Converged:',gradNorm<args.eps
    print 'Saving trained β in',args.beta
    writeBeta(args.beta,beta)

if (args.testdata):
    # Read beta from args.beta, and evaluate its MSE over data
    print 'Reading test data from',args.testdata
    data = readData(args.testdata,sc)
    data = data.repartition(args.N).cache()

    print 'Reading beta from',args.beta
    beta = readBeta(args.beta)

    print 'Computing MSE on data',args.testdata
    MSE = F(data,beta)
    print 'MSE is:', MSE
