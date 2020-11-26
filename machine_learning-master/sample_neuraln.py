import numpy as np
from numpy import random

def sigmoid(x):
    return 1/ (1+np.exp(-x))

def sig_der(x):
    return x * (1 - x)

training_inputs = np.array([[0,0,1],
                           [1,1,1],
                           [1,0,1],
                           [0,1,1]])

training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3,1)) - 1

print("Random starting synaptic weights:\n",synaptic_weights)

for i in range(100):
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    error = training_inputs - outputs
    adjustments = error * sig_der(outputs)
    synaptic_weights += np.dot(input_layer.T, adjustments)

print("Synaptic weights after training:\n", synaptic_weights)
print("Outputs after training: \n", outputs)

