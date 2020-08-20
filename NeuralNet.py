import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing


class NeuralNet:
    def __init__(self, train, header = True, h1 = 4, h2 = 2):
        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers
        raw_input = pd.read_csv(train)
        train_dataset = self.preprocess(raw_input)
        ncols = len(train_dataset.columns)
        nrows = len(train_dataset.index)
        self.X = train_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        self.y = train_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))

   
    def __activation(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid(self, x)
        elif activation == "tanh":
            self.__tanh(self, x)
        elif activation == "ReLu":
            self.__ReLu(self, x)

    
    def __activation_derivative(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)
        elif activation == "tanh":
            self.__tanh_derivative(self, x)
        elif activation == "ReLu":
            self.__ReLu_derivative(self, x)

    
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def __tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    def __ReLu(self, x):
        return np.maximum(0,x)

    
    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    def __tanh_derivative(self, x):
        return 1 - (x*x)
    def __ReLu_derivative(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x


    def preprocess(self, X):
        X = X.dropna()
        X = X.drop_duplicates()
        label_encoder = LabelEncoder()
        for i in range(X.shape[1]):
            X[X.columns[i]] = label_encoder.fit_transform(X[X.columns[i]])
        X = preprocessing.normalize(X)
        X = pd.DataFrame(X)
        return X
    
    # Below is the training function
    # We should change the max_iterations and learning_rate for tuning the network to produce better results
    def train(self,actf, max_iterations = 1000, learning_rate = 0.05):   
        for iteration in range(max_iterations):
            out = self.forward_pass(actf)
            error = 0.5 * np.power((out - self.y), 2)
            if actf=="sigmoid":
                self.backward_pass(out, activation="sigmoid")
            elif actf=="tanh":
                self.backward_pass(out, activation="tanh")
            elif actf=="ReLu":
                self.backward_pass(out, activation="ReLu")
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input

        print("After " + str(max_iterations) + " iterations with learning rate "+ str(learning_rate) +" the total error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)
        print(self.w23)


    def forward_pass(self,actf):
        # passing inputs through neural network
        if actf=="sigmoid":
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__sigmoid(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__sigmoid(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__sigmoid(in3)
        elif actf=="tanh":
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__tanh(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__tanh(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__tanh(in3)
        elif actf=="ReLu":
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__ReLu(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__ReLu(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__ReLu(in3)
        return out

    def backward_pass(self, out, activation):
        # passing inputs through neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)

    def compute_output_delta(self, out, activation):
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        elif activation == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))
        elif activation == "ReLu":
            delta_output = (self.y - out) * (self.__ReLu_derivative(out))
        self.deltaOut = delta_output

    def compute_hidden_layer2_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))
        elif activation == "tanh":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))
        elif activation == "ReLu":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__ReLu_derivative(self.X23))
        self.delta23 = delta_hidden_layer2

    def compute_hidden_layer1_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
        elif activation == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))
        elif activation == "ReLu":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__ReLu_derivative(self.X12))

        self.delta12 = delta_hidden_layer1

    def compute_input_layer_delta(self, activation):
        if activation == "sigmoid":
            delta_input_layer = np.multiply(self.__sigmoid_derivative(self.X01), self.delta01.dot(self.w01.T))
        elif activation == "tanh":
            delta_input_layer = np.multiply(self.__tanh_derivative(self.X01), self.delta01.dot(self.w01.T))
        elif activation == "ReLu":
            delta_input_layer = np.multiply(self.__ReLu_derivative(self.X01), self.delta01.dot(self.w01.T))

        self.delta01 = delta_input_layer

    
    def predict(self,test,tt, header = True):
        raw_input = pd.read_csv(test)
        test_dataset = self.preprocess(raw_input)
        tncols = len(test_dataset.columns)
        tnrows = len(test_dataset.index)
        self.X = test_dataset.iloc[:, 0:(tncols -1)].values.reshape(tnrows, tncols-1)
        self.y = test_dataset.iloc[:, (tncols-1)].values.reshape(tnrows, 1)
        if tt=="sigmoid": 
            tout=self.forward_pass("sigmoid")
        elif tt=="tanh":
            tout=self.forward_pass("tanh")
        elif tt=="ReLu":
            tout=self.forward_pass("ReLu")
        terror = 0.5 * np.power((tout - self.y), 2)
        t_error=np.sum(terror)
        return t_error


if __name__ == "__main__":
    neural_network = NeuralNet("brestcancer_train.csv")
    f="sigmoid"      # We should specify the activation function here.
    # The activation function can be either sigmoid or tanh or ReLu.
    neural_network.train(f)
    testError = neural_network.predict("brestcancer_test.csv",f) 
    print("testError with '" + str(f) + "' activation function : "+ str(testError))
