from datasets import *
from nn import *

def main():

    #create a dict where keys are 'x', 'y', and 'out'
    xor_data = xor_dataset()
    print("\nXOR dataset: ")
    print(xor_data)

    #1 -> number of layers
    #2 -> input dimensions
    #3 -> output dimensions
    #4 -> hidden units in each layer
    #5 -> learning rate
    nn = create_nn(1, 2, 3, 4, 5)
    print("\nNeural network: ")
    print(nn)
    
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


if __name__ == "__main__":
    main()