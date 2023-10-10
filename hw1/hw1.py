import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    
def relu(x):
    """Implements ReLU (y=x for x>0)"""
    return x*(x > 0)

def d_relu(x):
    """Implements d/dx[ReLU(x)]"""
    return 1*(x>0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(-x))

def forward_pass(x, w1, w2, b1, b2):
    """Inputs - x: 5x1 np array"""
    # z1 = w1@x + b1.T
    x.shape = (-1, 1)
    z1 = np.dot(x.T, w1.T) + b1.T
    a1 = sigmoid(z1)

    # z2 = w2@a1.T + b2
    z2 = np.dot(a1, w2.T) + b2.T
    # print(z2)
    # print(w2)
    # print(a1.T)
    # print(w2@a1.T)
    a2 = sigmoid(z2)
    return z1, z2, a1, a2

def error(y, y_hat):
    # err1 = np.mean(np.square(np.subtract(y_hat[0], y[0])))
    # err2 = np.mean(np.square(np.subtract(y_hat[1], y[1])))
    err = np.mean(np.square(np.subtract(y_hat, y)))
    return err

def d_error(y, y_hat):
    dE_da1 = 2*np.subtract(y_hat[0], y[0])
    dE_da2 = 2*np.subtract(y_hat[1], y[1])
    ret = np.empty((2,1))
    ret[0] = dE_da1
    ret[1] = dE_da2
    return ret

def calc_delta(dE_da, a):
    da_dz = np.multiply(a, (1-a))
    delta = np.multiply(dE_da, da_dz)
    return delta

def update_weights(w, weight_gradient, learning_rate):
    new_weights = np.subtract(w, learning_rate*weight_gradient)
    return new_weights

def update_bias(b, bias_gradient, learning_rate):
    new_bias = np.subtract(b, learning_rate*bias_gradient)
    return new_bias

def export_if_good(w1, w2, b1, b2):
    pass

def train(input_data, output_data, num_hidden):
    # number of inputs and outputs
    n = len(output_data[0])
    m = len(input_data[0])
    # initialize random weights and biases
    w1 = np.random.rand(num_hidden, m)
    w2 = np.random.rand(n, num_hidden)
    b1 = np.random.rand(num_hidden, 1)
    b2 = np.random.rand(n, 1)
    # initialize vars
    err_list = []
    count_correct_list = []
    learning_rate = 0.1
    for epoch in tqdm(range(5), desc="Iterating..."):

        for i, x in enumerate(input_data):
            # x.shape = (-1, 1)
            
            y = output_data[i]

            # foraward pass and error
            z1, z2, a1, a2 = forward_pass(x, w1, w2, b1, b2)
        
            err = error(y, a2)
            err_list.append(err)

            # print(i)
            # print(z1)
            # print(z2)
            # print(y)

            d_err = 2*(np.subtract(a2, y))

            # step 0: calculate the 'delta' term for layer 2 (da2/dz2*dE/da2)
            # delta = calc_delta(a2, d_err.T)
            delta = d_err * (sigmoid(z2)*(1-sigmoid(z2)))

            # step 1: dE/dW and dE/dB for layer 2
                # how much the error depends on w2 and b2
            # weight_gradient_l2 = delta@a1
            weight_gradient_l2 = delta.T*a1
            bias_gradient_l2 = delta
            
            # step 2: dE/da_1
                # how much the error depends on a1
            activation_gradient_l1 = w2.T@delta.T

            # print(activation_gradient_l1)
            
            # step 3: dE/dW and dE/dB for layer 1
                # how much the error depends on w1 and b1
            # da1_dz1 = np.multiply(a1, (1-a1))
            da1_dz1 = (sigmoid(z1)*(1-sigmoid(z1)))
            # print(da1_dz1)
            # print(activation_gradient_l1)
            # da1_dz1 = d_relu(z1)    ### CHANGE IF USING SIGMOID IN FORWARD PASS
            weight_gradient_l1 = x.T*(np.multiply(da1_dz1.T, activation_gradient_l1))
            # print(weight_gradient_l1)
            bias_gradient_l1 = np.multiply(da1_dz1.T, activation_gradient_l1)

            w1 = update_weights(w1, weight_gradient_l1, learning_rate)
            b1 = update_bias(b1, bias_gradient_l1, learning_rate)
            w2 = update_weights(w2, weight_gradient_l2, learning_rate)
            b2 = update_bias(b2, bias_gradient_l2.T, learning_rate)

        count_correct = 0
        for j, x in enumerate(input_data):
            # foraward pass
            z1, z2, a1, a2 = forward_pass(x, w1, w2, b1, b2)
            y = output_data[j]

            if np.array_equal(np.round(a2), y.reshape((1,2))):
                count_correct += 1

        # print(f"Test accuracy: {count_correct/400}")

        count_correct_list.append(count_correct)
        lr_decay = 1
        learning_rate *= (1. / (1. + lr_decay * epoch))
    return err_list, epoch, count_correct_list

####### flight code

path = "train1.csv"
inputs, outputs = preprocess(path)

err_list, epoch, count_correct = train(inputs, outputs, num_hidden=3)
iteration_list = np.linspace(0, epoch, len(count_correct))

fig, ax = plt.subplots(1,1)
ax.plot(iteration_list, np.array(count_correct)/400)
plt.show()






