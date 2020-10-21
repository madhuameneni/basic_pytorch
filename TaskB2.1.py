#relu ----
import torch
import torch.nn as nn

X = torch.tensor(([2, 9], [1, 5], [3, 6]), dtype=torch.float) # 3 X 2 tensor
y = torch.tensor(([92], [100], [89]), dtype=torch.float) # 3 X 1 tensor
xPredicted = torch.tensor(([4, 8]), dtype=torch.float) # 1 X 2 tensor

print(X.size())
print(y.size())

# scale units
X_max, _ = torch.max(X, 0) # gets the max value example 2,1,3 -> 3 , 9,5,6 -> 9
print("madhu x_max : " + str(X_max));
xPredicted_max, _ = torch.max(xPredicted, 0)
print("madhu xxxx : " + str(xPredicted_max));
X = torch.div(X, X_max)
print("madhu xx : " + str(X));
xPredicted = torch.div(xPredicted, X_max)
y = y / 100  # max test score is 100

test_x = torch.tensor(( [1, 5] ), dtype=torch.float) # 1 X 2 tensor
test_y = torch.tensor(( [100] ), dtype=torch.float) # 1 X 1 tensor
test_x = torch.div(test_x, X_max)
test_y = test_y / 100

class Neural_Network(nn.Module):
    def __init__(self, ):
        super(Neural_Network, self).__init__()
        # parameters
        # TODO: parameters can be parameterized instead of declaring them here
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        self.relu = nn.ReLU(inplace=True)
        
        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize) # 2 X 3 tensor
        self.W2 = torch.randn(self.hiddenSize, self.outputSize) # 3 X 1 tensor
        
    def forward(self, X):
        self.z = torch.matmul(X, self.W1) # 3 X 3 ".dot" does not broadcast in PyTorch
        self.z2 = self.relu(self.z) # activation function
        print("Activation relu : " + str(self.z2));
        self.z3 = torch.matmul(self.z2, self.W2)
        o = self.relu(self.z3) # final activation function
        return o
    
    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)
    
    def backward(self, X, y, o):
        self.o_error = y - o # error in output
        self.o_delta = self.o_error * self.sigmoidPrime(o) # derivative of sig to error
        self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        self.W1 += torch.matmul(torch.t(X), self.z2_delta)
        self.W2 += torch.matmul(torch.t(self.z2), self.o_delta)
        
    def train(self, X, y):
        # forward + backward pass for training
        o = self.forward(X)
        self.backward(X, y, o)
        
    def test(self, X, y):
        # Forward step - predict
        y_pred = self.forward(X)
        # Compute and print loss
        loss = y_pred - y;
        print("Loss for Test: " + str(loss.item()) )
        print("True output:" + str(y) + "predicted: " + str(y_pred) )
        
    def saveWeights(self, model):
        # we will use the PyTorch internal storage functions
        torch.save(model, "NN")
        # you can reload model with all the weights and so forth with:
        # torch.load("NN")
        
    def predict(self):
        print ("Predicted data based on trained weights: ")
        print ("Input (scaled): \n" + str(xPredicted))
        print ("Output: \n" + str(self.forward(xPredicted)))

NN = Neural_Network()
for i in range(200):  # trains the NN 1,000 times
    print ("#" + str(i) + " Loss: " + str(torch.mean((y - NN(X))**2).detach().item()))  # mean sum squared loss
    NN.train(X, y)
NN.saveWeights(NN)
NN.test(test_x, test_y);
NN.predict()
