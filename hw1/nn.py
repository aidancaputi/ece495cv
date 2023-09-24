import math, random
from forward_mode import *

#multiplies input vector and weights
def dot(u, v):

    #print("doing dot product of " + str(u) + " and " + str(v))

    #make sure the two vectors are going to work together in terms of dimensions
    if(len(u) != len(v[0])):
        print("Error in matmul() -> " + str(u) + " and " + str(v) + ", their dimensions don't match!")
        exit(1)

    output = []
    
    #for each row in weights matrix
    for i in range(len(v)):
        col_sum = 0

        #go through input vec 
        for j in range(len(u)):

            #multiply the vector element times the correspending element in the matrix row
            col_sum += (u[j]* v[i][j])

        #add the column total to the output vector
        output.append(col_sum)

    #print("the dot product was " + str(output))

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
        print("Error in mlp() -> not the same amount of weights and biases!")
        exit(1)
    
    out = x

    #loop through weights and bias and apply nested single layer perceptron starting from the first
    for w, b in zip(w_vec, b_vec):
        out = slp(out, w, b)

    return out

#scale a vector element-wise by a constant
def ktimesv(k, u): 
    return [k*u[i] for i in range(len(u))]

def vector_subtraction(u, v):
    if(len(u) != len(v)):
        print()
    output = []
    for i in range(len(u)):
        output.append(u[i] - v[i])
    return output

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
            row.append(random.uniform(-1, 1))
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

def split(X, y, train_test_split):

    #make sure X and y data have matching sizes
    if(len(X) != len(y)):
        print("ERROR: X and y are not the same size. Check your training data to make sure every X has a y.")
        exit(1)

    X_train = X
    y_train = y
    X_test = []
    y_test = []

    num_samples = len(X)

    train_size = int(train_test_split * num_samples)
    test_size = num_samples - train_size

    for i in range(test_size):

        #randomly choose an index in X_train
        chosen_idx = random.randint(0, len(X_train) - 1)

        #pop the X_train and y_train values at that index 
        popped_X = X_train.pop(chosen_idx)
        popped_y = y_train.pop(chosen_idx)

        #add the popped values to the test data
        X_test.append(popped_X)
        y_test.append(popped_y)

    return X_train, y_train, X_test, y_test

#loss function
def squared_loss(u, v):
    
    # (u - v) * (u - v)
    #note here we must make the second (u - v) an element of a list so that the dot function can work properly
    #i.e., the dot function has to think that the second vector is actually a matrix
    #there is probably a smarter way around this but i am lazy
    return dot(vector_subtraction(u, v), [vector_subtraction(u, v)])

def squared_loss_gradient(weights, X, y):
    
    return ktimesv(-2, dot(vector_subtraction(y, dot(X, weights)), X))

#train the network
def train(X, y, n_layers, input_dim, output_dim, hidden_units, learning_rate, train_test_split):

    #log the details of this training
    print_train_status(X, y, n_layers, input_dim, output_dim, hidden_units, learning_rate, train_test_split)

    #use hyperparameters to initialize weights for the network
    weights = initialize_weights(n_layers, hidden_units, input_dim, output_dim)
    print(*weights, sep='\n')

    #use hyperparameters to initialize biases to 0
    biases = initialize_biases(n_layers, hidden_units, output_dim)
    print(*biases, sep='\n')

    X_train, y_train, X_test, y_test = split(X, y, train_test_split)

    print("training X: ", X_train)
    print("training y: ", y_train)
    print("test X: ", X_test)
    print("test y: ", y_test)

    #for each training X
    for cur_X, cur_y in zip(X_train, y_train):

        #compute the loss between its actual y and the MLP output
        #print("\ncomputing loss between " + str(cur_y) + " and " + str(mlp(cur_X, weights, biases)))
        cur_loss = squared_loss(cur_y, mlp(cur_X, weights, biases))
        print("\nloss for " + str(cur_X) + " was: ", cur_loss)

        #gradient descent
        #weights = vector_subtraction(weights, ktimesv(learning_rate, squared_loss_gradient(weights, cur_X, cur_y)))


    return