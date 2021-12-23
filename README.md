# Nuwa_framework
A deep learning framework powered by Zixing by using only numpy, aims to build a deep learning framework to compute 
partial differential equations with higher performance. This project is still in early development stage, and the 
underlying development of fully connected neural network and part of 2D CNN neural network has been completed. 
A dynamic back propagation graph is provided, it supports all latest optimizer, normalizations. 
The second phase will improve the project of the first phase to adapt it to the needs of computing partial differential 
equations. Finally, the Nuwa framework will be deployed as a C++ project and support CUDA and parallel computing. 
If you are interested in approving the deep learning framework calculation efficiency, you can contact me by my mail: 
zixing.qiu@etu.toulouse-inp.fr


In this folder, you'll find:
<br>
<br>
  **1 Nuwa.ipynb:** A demo contains all the module of Nuwa framework, it contains 4 main parts of which the architecture 
will be represent later. It also provide a testing set, 
  you could try your own dataset following the guide.
<br>  
  **2 Nuwa v0.0.1.png:** Nuwa v0.0.1's architecture, **Black bold** unit represents it's a python class while green 
dashed line represents the relationship between these classes. For       example, **Dataset** class parse the dataset to
**NN**. Details are in the following section.

  **3 demo.py:** Provide the same demo of two examples as Nuwa.ipynb.

Before start, use "pip install -r requirements.txt" to install all the requirements

## <div align="center">Architecture</div>
The Nuwa framework consists of four parts in total: **Dataset**, **NN**, **Optimizer** and **Visual**. **Dataset** is 
responsible for dataset allocation, pre-processing and IO, **NN** contains different kinds of neural network layer 
such as fully connected layer, convolution layer etc, it also provide activation functions, it could be aclass which 
inherite Layer class, or it can be contained in other layers like fully connected layer and be calculated at the end 
of layer. **Optimizer** is a kind of "kernel" function of Nuwa. Inside, it provides optimizers like gradient descent 
and SGD, also it provides training function for training and calculate backpropagation by using dynamic computation 
graphe, each node in the graph is a **Node** type. It enables chain rule calculation be more efficient. You can always 
moniter gradient flow to check if there is gradient vanish during training. At last, **Visual** provides a series graph 
tool for plotting loss list or the gradient flow.

    for using these four parts, import as following:
    
    from dataloader import Dataset
    
    from network import NN
    
    from optimizers import Optimizer

from visualization import Visual
## <div align="center">Dataset class</div>
By default, Dataset take X,y,proportion=0.8,shuffle=True, mini_batch=0 as variables and params. We assume that X is the 
feature training data and y its respecting label, they have same lines and can be concatenated by their axis 1. 
Also you can distribute dataset into training and test dataset by certain given proportion. During the mini batch 
graident descent training, it can also provide mini batchs by using builtin function getminiset(). It also provides 
shuffle methods, for example, in minibatch gradient descent, where each epoch can shuffle the dataset for avoiding 
optimizers parsing exactly the same data each epoch.


## <div align="center">NN class</div>
NN simply represents "neural network". Since we have initialized dataset, we take it as variable and parse it in **NN** 
class. With the help of Layer class defined by NN, we can create a deep learning network like in the demo:

    layer_list = [NN.Layer('Linear',3,10,'LeakyReLU'),NN.Layer('Linear',10,3,'LeakyReLU'), NN.Layer('Linear',3,1,'none')]

here take linear layer as example, **Layer** class takes type, input_dim, output_dim and activation as params, 
one can also use 

    LinearLayer(input_dim, output_dim) 

to initialize a linear layer, defaultly, type equals 'Linear' and activation is None.
These two methods are exactly the same except the latter automatically set the layer type as linear.

ActivationFunc is a class which contains several static activation function methods: sigmoid, ReLU, LeakyReLU, 
tanh or none. it takes features from the last layer. You also have two methods to use activation functions. One is 
defined inside **Layer** class. Or, **NN** provide activation function as **Layer**.

Aim of this architecture design is that, this allows the freedom to organize the architecture of the neural network, 
allowing to try different batch normalization positions to achieve a better fit.

All variables are as follows: type, input_dim, output_dim, activation, BN = False. 
The last variable BN accepts a Boolean argument, if True, batch normalization will be performed and two complete 
examples are provided in **Nuwa.ipynb**.

## <div align="center">Optimizer class</div>
**Optimizer** is the most important class in the architecture of Nuwa, as the training process and testing process 
are provided by it:

1 loss function

2 train(): including forword propagation and backpropagation

3 **Node** class allows every calculation in the backpropagation can be updated. 

4 lists: gradient list, weights list, loss list and passer list. All the list is accessible during model training

There are several options of loss function, v0.0.1 provide Logarithmic and Quadratic two basic functions; A upcoming 
version will provide PDE 
typical loss like MSE etc 

    loss_func = Optimizer.LossFunc.Quadratic

Every loss function has 2 functions: loss() and diff(), loss() returns loss while diff() returns derivative corresponds 
input data and loss function
Use following line as Optimizer init:

    optim = Optimizer(nn,"SGD",iter = 20000, lr=1e-6),

if you are using mini batch gradient descent, use API as: 

    optim = Optimizer(nn,"minibatchgd",epoch = 1000, lr=1e-4, decay_rate=0.01)

nn is an **NN** object, "SGD" is the optimize method to use, there are several options for now: "GD", "SGD", 
"minibatchgd". epoch = epoch numbers and lr the learning rate.

For testing the trained weight on the testset or other feature data X, use following line:
    
    optim.train()
    optim.predict(X)

## <div align="center">Visual class</div>
Version 0.0.1 provides plotloss() and plotgradientnorm() to plot loss flow and gradient flow. More plots available in 
very soon

Thus the whole training example is given as:

    layer_list = [NN.Layer('Linear',3,10,'sigmoid',BN=True), NN.Layer('Linear',10,100,'sigmoid',BN=True),
                  NN.Layer('Linear',100,10,'sigmoid',BN=True),NN.Layer('Linear',10,3,'none') ]
    
    dataset = Dataset(X, y, mini_batch= 64)
    
    nn = NN(dataset)
    
    nn.addlayers(layer_list)
    
    loss_func = Optimizer.LossFunc.Quadratic
    
    optim = Optimizer(nn,"minibatchgd", loss_func, epoch = 1000, lr=1e-4, decay_rate=0.01)
    
    optim.train()
    
    visual = Visual(optim)
    
    visual.plotloss()
    
    visual.plotgradientnorm()
