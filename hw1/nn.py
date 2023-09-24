import math, random

#multiplies input vector and weights
def dot(input_vec, weights_matrix):

    #make sure the two vectors are going to work together in terms of dimensions
    if(len(input_vec) != len(weights_matrix)):
        raise Exception("Error in matmul() -> " + input_vec + " and " + weights_matrix + ", their dimensions don't match!")

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
    dot_prod = dot(x, w)

    #add bias to each value in the resulting vector
    for i in range(len(dot_prod)):
        output.append(dot_prod[i] + b[i])

    return output

#sigmoid function element wise to a vector
def sigmoid(vec):
    return [(1/(1 + (math.exp(-1 * x)))) for x in vec]

#single layer perceptron
def slp(x, w, b):
    return sigmoid(fc_linear_layer(x, w, b))

#multi-layer perceptron
def mlp(x, w_vec, b_vec):

    #make sure weights and biases vectors are same length
    if(len(w_vec) != len(b_vec)):
        raise Exception("Error in mlp() -> not the same amount of weights and biases!")
    
    out = x

    #loop through weights and bias and apply nested single layer perceptron starting from the first
    for w, b in zip(w_vec, b_vec):
        out = slp(out, w, b)

    return out

#scale a vector element-wise by a constant
def ktimesv(k, u): 
    return [k*u[i] for i in range(len(u))]

#vector addition
def vplus(u, v): 
    return [u[i]+v[i] for i in range(len(u))]

#vector subtraction
def vminus(u, v): 
    return vplus(u, ktimesv(-1, v))

#loss function
def loss(u, v):

    # (u - v) * (u - v)
    return dot(vminus(u, v), vminus(u, v))

#take the hyperparameters and initialize the weights
def initialize_weights(n_layers, hidden_units, input_dim, output_dim):

    weights = []

    for i in range(n_layers):

        #this is the input layer weights
        if(i == 0):
            weights.append(generate_random_matrix(hidden_units, input_dim))
        
        #this is a middle hidden layer (i.e., not the output layer)
        elif((i > 0) and (i < (n_layers - 1))):
            weights.append(generate_random_matrix(hidden_units, hidden_units))

        #this is the output layer
        else:
            weights.append(generate_random_matrix(output_dim, hidden_units))
    
    return weights

#generates a matrix with random numbers between 0 and 1
def generate_random_matrix(n_rows, n_cols):
    matrix = []
    for i in range(n_rows):
        row = []
        for j in range(n_cols):
            row.append(random.random())
        matrix.append(row)
    return matrix

def initialize_biases(n_layers, hidden_units, output_dim):
    biases = []

    #for each layer
    for i in range(n_layers):
        
        #create bias vector with same size as output of that layer
         #this is the input layer weights
        if(i < (n_layers - 1)):
            biases.append([0] * hidden_units)

        #this is the output layer
        else:
            biases.append([0] * output_dim)

    return biases

#simply prints all the info about the network being trained
def print_train_status(X, y, n_layers, input_dim, output_dim, hidden_units, learning_rate, train_test_split):

    print("\nTraining MLP network using the following data:")
    print("X data: ", X)
    print("y data: ", y)

    print("\nAnd the following parameters:")
    print("Number of layers: ", n_layers)
    print("Input dimensions: ", input_dim)
    print("Output dimensions: ", output_dim)
    print("Number of hidden units in each layer: ", hidden_units)
    print("Learning rate: ", learning_rate)
    print("Train/test split: ", train_test_split)

    return

def train(X, y, n_layers, input_dim, output_dim, hidden_units, learning_rate, train_test_split):

    #log the details of this training
    print_train_status(X, y, n_layers, input_dim, output_dim, hidden_units, learning_rate, train_test_split)

    #use hyperparameters to initialize weights for the network
    weights = initialize_weights(n_layers, hidden_units, input_dim, output_dim)
    print(*weights, sep='\n')

    #use hyperparameters to initialize biases to 0
    biases = initialize_biases(n_layers, hidden_units, output_dim)
    print(*biases, sep='\n')

    return