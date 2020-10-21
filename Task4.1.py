import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


X1_data = pd.read_csv('abs_input.csv')
Y1_data = pd.read_csv('abs_output.csv')

x1 = torch.tensor(np.array(X1_data), dtype=torch.float)
y1 = torch.tensor(np.array(Y1_data), dtype=torch.float)

##X3_data = pd.read_csv('5-5-20/Freq/Set1/impinging_tri-tone2.csv')
##y3_data = pd.read_csv('5-5-20/Freq/Set1/output_tri-tone2.csv')

print(x1.size())
print(y1.size())

##x = torch.tensor(([2, 9], [1, 5], [3, 6]), dtype=torch.float) # 3 X 2 tensor
##y = torch.tensor(([92], [100], [89]), dtype=torch.float) # 3 X 1 tensor
##
##xPredicted = torch.tensor(([4, 8]), dtype=torch.float) # 1 X 2 tensor
##
##print(x.size())
##print(y.size())
##
### scale units
##X_max, _ = torch.max(x, 0)
##xPredicted_max, _ = torch.max(xPredicted, 0)
##
##x = torch.div(x, X_max)
##xPredicted = torch.div(xPredicted, xPredicted_max)
##y = y / 100  # max test score is 100
##
##test_x = torch.tensor(( [1, 5] ), dtype=torch.float) # 1 X 2 tensor
##test_y = torch.tensor(( [100] ), dtype=torch.float) # 1 X 1 tensor
##test_x = torch.div(test_x, X_max)
##test_y = test_y / 100

class Neural_Network(nn.Sequential):
    def __init__(self):
        super(Neural_Network, self).__init__()
        # TODO: parameters can be parameterized instead of declaring them here
        self.inputSize = 3073
        self.outputSize = 2048
        self.hiddenSize = 10000
        print("madhughjgj" )
        ## self.seq = nn.Sequential(torch.nn.Sigmoid(),torch.nn.Linear(self.inputSize, self.hiddenSize),torch.nn.Sigmoid(),torch.nn.Linear(self.hiddenSize, self.outputSize),torch.nn.Sigmoid(),torch.nn.Linear(self.hiddenSize, self.outputSize), torch.nn.Sigmoid(),)
        ##            torch.nn.ReLU(),
          
        ##            torch.nn.Softsign(),
        ##            torch.nn.Tanh(),
        ##            torch.nn.Tanhshrink(),
        #torch.nn.Linear(self.hiddenSize, self.outputSize),\
        # torch.nn.Sigmoid(),)
        # parameters
        self.seq = torch.nn.Sequential(torch.nn.Linear(self.inputSize, self.hiddenSize), \
                   torch.nn.Sigmoid(), \
                   torch.nn.Linear(self.hiddenSize, 8000),\
                   torch.nn.Sigmoid(), \
                   torch.nn.Linear(8000, 6000), \
                   torch.nn.Sigmoid(), \
                   torch.nn.Linear(6000, 4000), \
                   torch.nn.Sigmoid(), \
                   torch.nn.Linear(4000, self.outputSize),)
        print("madhu" , self.seq)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.SGD(self.seq.parameters(), lr=1e-4)
        # Another optimizer setup - with MOMENTUM for non-trivial data/relationship
        #self.optimizer = torch.optim.SGD(self.seq.parameters(), lr=1e-4, momentum=0.9)


    def train(self, X, y, t):
        # Forward step - predict
        print(X.size())
        y_pred = self.seq(X)
        # Compute and print loss
        loss = self.criterion(y_pred, y)
        if t % 1000 == 9:
            print(t, loss.item())
##            plt.plot(y_pred[10,:].detach(), label="y_pred")
##            plt.plot(y[10,:], label="y"+str(t))
##            plt.legend()
 #           plt.draw()
 #           plt.pause(0.001)
##            plt.show(block=False);

        # Zero gradients, perform a backward pass, and update the weights.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test(self, X, y):
        # Forward step - predict
        y_pred = self.seq(X)
        # Compute and print loss
        loss = self.criterion(y_pred, y)
        #logy = y.log();
        plt.plot(y, label="final y")
        print(" y data: ", y )
        print(" for size: ", y.size() )
        plt.plot(y_pred.detach(), label="final prediction")
        plt.legend()
 #       plt.save()
        plt.show()
        print("Loss for Test: " + str(loss.item()) )
        print("True output:" + str(y) + "predicted: " + str(y_pred) )

 
    def test3D(self, X, y):
        print("Loss for Testdfsdf: " + str(self.inputSize) )
        # Forward step - predict
        y_pred = self.seq(X)
        # Compute and print loss
        loss = self.criterion(y_pred, y)
        #logy = y.log();
        ##        plt.imshow(y, label="final y")
        ##        plt.imshow(y_pred.detach(), label="final prediction")
        ##        plt.legend()
        ##        plt.draw()
        ##        plt.pause(10)
        ##        plt.clf()
        ##        plt.close()
        #       plt.save()
        #        plt.show()
        print("Loss for Test: " + str(loss.item()) )
        print("True output:" + str(y) + "predicted: " + str(y_pred) )
        
    def saveWeights(self, model):
        # we will use the PyTorch internal storage functions
        torch.save(model, "NN")
        # you can reload model with all the weights and so forth with:
        # torch.load("NN")
        
    def predict(self, in_data):
        return self.seq(in_data)

NN = Neural_Network()
##plt.imshow(y1)
##plt.show()
test_x = x1[10, :]
test_y = y1[10, :]
##log_ty =test_y.log10().
##plt.plot(test_x, label="x")
##plt.plot(test_y, label="y")
##plt.show()
TRAIN=1
if (TRAIN==1):
    print("Starting..")
    for i in range(100):  # trains the NN 10,000 times
        NN.train(x1, y1, i)
    NN.saveWeights(NN)
else:
    print("Loading NN...")
    NN = torch.load("NN")
        
#NN.test3D(x1, y1)

y1_pred = NN.predict(x1)
e=y1-y1_pred
##plt.imshow(e.detach())
print(e.min(), e.max())
NN.test(test_x, test_y);
##NN.predict()