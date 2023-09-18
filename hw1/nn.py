#created a dictionary with network information
def create_nn(n_layers, in_dim, out_dim, hl_size, learning_rate):
    nn = {}
    nn.update({'n_layers':n_layers})
    nn.update({'in_dim':in_dim})
    nn.update({'out_dim':out_dim})
    nn.update({'hl_size':hl_size})
    nn.update({'learning_rate':learning_rate})
    return nn

#multiplies input vector and weights
def matmul(input_vec, weights_matrix):

    #make sure the two vectors are going to work together in terms of dimensions
    if(len(input_vec) != len(weights_matrix)):
        return "Error multiplying " + input_vec + " and " + weights_matrix + ", their dimensions don't match!"

    output = []
    
    #for each column in weights matrix
    for i in range(len(weights_matrix[0])):
        col_sum = 0

        #go through input vec
        for j in range(len(input_vec)):

            #multiply each input vec value with the corresponding index of column of the matrix
            col_sum += (input_vec[j] * weights_matrix[j][i])

        #add the column total to the output vector
        output.append(col_sum)

    return output

#multiple input and weights, then add bias
def fc_linear_layer(x, w, b):
    
    output = []

    #multiply input with weights
    dot = matmul(x, w)

    #add bias to each value in the resulting vector
    for i in range(len(dot)):
        output.append(dot[i] + b[i])

    return output

#sigmoid function element wise to a vector
def sigmoid(vec):
    e = 2.718281828459045
    return [(1/(1 + (e**(-1 * x)))) for x in vec]

#single layer perceptron
def slp(x, w, b):
    return sigmoid(fc_linear_layer(x, w, b))

#multi-layer perceptron
def mlp(x, w_vec, b_vec):
    return