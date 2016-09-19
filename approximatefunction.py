from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection, RecurrentNetwork
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer
import matplotlib.pyplot as plt
import sys
import math
import numpy

"graphs a goal function and then trains a neural net to approximate it"

def graphfunction(xmin,xmax,xres,function,*args):
    "takes a given mathematical function and returns a set of points to graph"

    x,y = [],[]
    i=0
    while xmin+i*xres<=xmax:
        x.append(xmin+i*xres)
        y.append(function(x[i],*args))
        i+=1
    return [x,y]

def parsefunction(x, functionstring):
    "takes a string and parses it into a function"

    #if it's in a y= format, get rid of that
    if(len(functionstring) > 2 and functionstring[1] == '='):
            functionstring = functionstring[2:]

    #look for multiplication and be sure the operator is there
    for i in range(len(functionstring)):
        if(functionstring[i] == 'x' and i > 0):
            if(functionstring[i-1].isdigit()):
                functionstring = functionstring[:i] + "*" + functionstring[i:]

    #will use eval (although not ideal) to evaluate the expression
    #may want to revisit this later and polish it up
    #need to get it to work for powers as well
    return eval(functionstring)

#set domain and resolution based on command line input
xmin = (float)(sys.argv[2])
xmax = (float)(sys.argv[3])
xres = (float)(sys.argv[4])
plt.xlim(xmin, xmax)


#build the net
#nn = buildNetwork()

#plot the goal function
[x,y] = graphfunction(xmin,xmax,xres,parsefunction,sys.argv[1])
plt.plot(x,y)

plt.show()
