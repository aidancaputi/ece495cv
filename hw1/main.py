from datasets import *
from nn import *

def main():

    #create a dict where keys are 'x', 'y', and 'out'
    X_xor = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]

    y_xor = [
        [0],
        [1],
        [1],
        [0],
    ]

    
    n_layers = 1 #number of layers
    input_dim = 2 #input dimensions
    output_dim = 3 #output dimensions
    hidden_units = 4 #hidden units in each layer
    learning_rate = 5 #learning rate
    
    #testing
    input_vec = [1,1]
    weights = [[1, 2],
               [3, 4]]
    bias = [3, 3]
    print("\nFully connected layer test: ")
    print("input: ", input_vec)
    print("weights: ", *weights, sep='\n')
    print("bias: ", bias)
    print("fully-connected linear layer output: ", fc_linear_layer(input_vec, weights, bias))
    print("single-layer perceptron output: ", slp(input_vec, weights, bias))
    weights_vec = [[[1, 2],[3, 4]], [[2, 3],[4, 5]]]
    bias_vec = [[2, 3], [3, 4]]
    print("multi-layer perceptron output: ", mlp(input_vec, weights_vec, bias_vec))

    train(X_xor, y_xor, n_layers, input_dim, output_dim, hidden_units, learning_rate)


if __name__ == "__main__":
    main()