from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection, RecurrentNetwork
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer

nn = FeedForwardNetwork(); #create the neural net

#construct input, hidden, output layers
inputLayer = LinearLayer(2)
hiddenLayer = SigmoidLayer(3)
outputLayer = LinearLayer(1)

#add the layers to the network
nn.addInputModule(inputLayer)
nn.addModule(hiddenLayer)
nn.addOutputModule(outputLayer)

#create connections between the different layers using full connectivity
input2hidden = FullConnection(inputLayer, hiddenLayer)
hidden2output = FullConnection(hiddenLayer, outputLayer)

#add those connections to the network
nn.addConnection(input2hidden)
nn.addConnection(hidden2output)

#sort the modules
nn.sortModules()

#recreate the neural network differently this time
nn = buildNetwork(2,3,1, bias=True, hiddenclass = TanhLayer)

#create the dataset
dataset = SupervisedDataSet(2,1) #2d input, 1d output

#add samples to the dataset specifying the XOR function
dataset.addSample((0,0),(0,))
dataset.addSample((0,1),(1,))
dataset.addSample((1,0),(1,))
dataset.addSample((1,1),(0,))

#create the trainer connecting the dataset and the neural net
trainer = BackpropTrainer(nn,dataset)


#print the input weights
print('Input Weights: ', input2hidden.params, '\n')


#train the network
print('training...\n')
for i in range(10000):
    trainer.train()

print("\n", nn.activate([1,0]))
print("\n", nn.activate([0,1]))
print("\n", nn.activate([1,1]))
print("\n", nn.activate([0,0]))
