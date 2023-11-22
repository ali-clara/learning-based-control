import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class NeuralNetwork():
    def __init__(self, num_inputs, num_outputs, num_hidden=10, weights=None, learning_rate=0.1, epochs=5):
        
        # initialize inputs and outputs
        n = num_outputs
        m = num_inputs

        self.input_data = None
        self.output_data = None

        # initialize random weights and biases
        self.w1 = np.random.rand(num_hidden, m)
        self.w2 = np.random.rand(n, num_hidden)
        self.b1 = np.random.rand(num_hidden, 1)
        self.b2 = np.random.rand(n, 1)

        if weights is not None:
            self.w1 = weights[0]
            self.w2 = weights[1]
            self.b1 = weights[2]
            self.b2 = weights[3]

        self.learning_rate = learning_rate
        self.epochs = epochs

    def set_data(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def d_sigmoid(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(-x))

    def error(self, y, y_hat):
        err = np.mean(np.square(np.subtract(y_hat, y)))
        return err

    def d_error(self, y, y_hat):
        ret = 2*(np.subtract(y_hat, y))
        return ret

    def calc_delta(self, dE_da, a):
        da_dz = np.multiply(a, (1-a))
        delta = np.multiply(dE_da, da_dz)
        return delta
    
    def forward_pass(self, x, test=False):
        """Inputs - x: Nx1 np array"""
        x.shape = (-1, 1)
        z1 = np.dot(x.T, self.w1.T) + self.b1.T
        a1 = self.sigmoid(z1)

        z2 = np.dot(a1, self.w2.T) + self.b2.T
        a2 = self.sigmoid(z2)

        if test:
            return a2[0]
        else:
            return z1, z2, a1, a2
    
    def backward_pass(self, x, d_err, a1, z1, z2):
        # step 0: calculate the 'delta' term for layer 2 (da2/dz2*dE/da2)
        delta = d_err * (self.sigmoid(z2)*(1-self.sigmoid(z2)))
        # step 1: dE/dW and dE/dB for layer 2
            # how much the error depends on w2 and b2
        weight_gradient_l2 = delta.T*a1
        bias_gradient_l2 = delta
        # step 2: dE/da_1
            # how much the error depends on a1
        activation_gradient_l1 = self.w2.T@delta.T
        # step 3: dE/dW and dE/dB for layer 1
            # how much the error depends on w1 and b1
        da1_dz1 = (self.sigmoid(z1)*(1-self.sigmoid(z1)))
        weight_gradient_l1 = x.T*(np.multiply(da1_dz1.T, activation_gradient_l1))
        bias_gradient_l1 = np.multiply(da1_dz1.T, activation_gradient_l1)

        self.update_weights(weight_gradient_l1, weight_gradient_l2, self.learning_rate)
        self.update_bias(bias_gradient_l1, bias_gradient_l2, self.learning_rate)

    def update_weights(self, weight_gradient_l1, weight_gradient_l2, learning_rate):
        self.w1 = np.subtract(self.w1, learning_rate*weight_gradient_l1)
        self.w2 = np.subtract(self.w2, learning_rate*weight_gradient_l2)

    def update_bias(self, bias_gradient_l1, bias_gradient_l2, learning_rate):
        self.b1 = np.subtract(self.b1, learning_rate*bias_gradient_l1)
        self.b2 = np.subtract(self.b2, learning_rate*bias_gradient_l2.T)

    def save_weights(self):
        np.save("w1.npy", self.w1)
        np.save("w2.npy", self.w2)
        np.save("b1.npy", self.b1)
        np.save("b2.npy", self.b2)

    def train(self):
        # initialize vars
        err_list = []
        # learning_rate = self.learning_rate
        for epoch in range(self.epochs):

            for i, x in enumerate(self.input_data):
                y = self.output_data[i]

                # foraward pass and error
                z1, z2, a1, a2 = self.forward_pass(x)
                err = self.error(y, a2)
                d_err = self.d_error(y, a2)
                err_list.append(err)

                # backward pass (gradient calculation and weight update)
                self.backward_pass(x, d_err, a1, z1, z2)

            lr_decay = 1
            self.learning_rate *= (1. / (1. + lr_decay * epoch))
        
        self.save_weights()
        return err_list

if __name__ == "__main__":
    def preprocess(data_path):
        """Returns - input_data: 400x5 np array of inputs (x)
                    output data: 200x5 np array of outputs (y)"""
        input_labels = ["x1", "x2", "x3", "x4", "x5"]
        output_labels = ["y1", "y2"]
        data = pd.read_csv(data_path, names=input_labels+output_labels)

        # create 400x5 numpy array of inputs
        input_data = np.zeros((len(data["x1"]), len(input_labels)))
        for i, label in enumerate(input_labels):
            # input_data[label] = data[label]
            input_data[:,i] = data[label]

        # create 400x2 numpy array for outputs
        output_data = np.zeros((len(data["y1"]), len(output_labels)))
        for i, label in enumerate(output_labels):
            output_data[:,i] = data[label]

        return input_data, output_data

    # load training and testing data
    pair = 1
    train_inputs, train_outputs = preprocess("train"+str(pair)+".csv")
    test_inputs, test_outputs = preprocess("test"+str(pair)+".csv")
    num_hidden = 10
    epochs = 5
    lr = 0.1

    model = NeuralNetwork(len(train_inputs[0]), len(train_outputs[0]), num_hidden, lr, epochs)
    model.set_data(train_inputs, train_outputs)
    error = model.train()

    # should decrease over iterations
    plt.plot(error)
    plt.show()




