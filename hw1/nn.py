import math, random

def dot(matrix1, matrix2):

    #print("doing dot prod with ", matrix1, matrix2)

    if((len(matrix1[0]) == len(matrix2[0])) and (len(matrix1) == len(matrix2))):
        matrix2 = transpose(matrix2)
    
    if len(matrix1[0]) != len(matrix2):
        print("wrong size matrices, can't dot product")
        exit(1)

    # Get the dimensions of the matrices
    m1 = len(matrix1)
    n1 = len(matrix1[0])
    m2 = len(matrix2)
    n2 = len(matrix2[0])

    #resulting matrix is 
    out = [[0]*n2 for i in range(m1)]

    for i in range(m1):
        for j in range(n2):
            for k in range(n1):
                out[i][j] += matrix1[i][k] * matrix2[k][j]

    #print("result was: ", out)

    return out

def transpose(matrix):
    #print("original matrix", matrix)
    orig_rows = len(matrix)
    orig_cols = len(matrix[0])
    
    transposed = [[0]*orig_rows for i in range(orig_cols)]

    for i in range(orig_rows):
        for j in range(orig_cols):
            transposed[j][i] = matrix[i][j]
    #print("transposed", transposed)
    return transposed

#multiple input and weights, then add bias
def fc_linear_layer(x, w, b):
    
    output = []

    #multiply input with weights
    dot_prod = dot(x, w)

    #print("dot was ", dot_prod)
    #print("bias are: ", b)

    #add bias to each value in the resulting vector
    for i in range(len(dot_prod)):
        new_row = []
        for j in range(len(dot_prod[i])):
            new_row.append(dot_prod[i][j] + b[i])
        output.append(new_row)

    #print("fc linear layer returned ", output)

    return output

#sigmoid function element wise to a vector
def sigmoid(vec):
    return [[(1/(1 + (math.exp(-1 * x)))) for x in tt] for tt in vec]

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
        #print("out ", out)

    return out

#scale a vector element-wise by a constant
def ktimesv(k, u): 
    #print("scaling k=", k, "to matrix:", u)

    scaled = []

    for i in range(len(u)):
        new_row = []
        for j in range(len(u[0])):
            new_row.append(u[i][j] * k)
        scaled.append(new_row)

    return scaled

def vector_subtraction(u, v):

    #print("subtracting ", u, v)
    if(len(u) != len(v)):
        print()
    output = []
    for i in range(len(u)):
        new_row = []
        for j in range(len(u[i])):
            new_row.append(u[i][j] - v[i][j])
        output.append(new_row)
    #print("subtraction was ", output)
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
    return dot(vector_subtraction(u, v), vector_subtraction(u, v))

#this must return something that is the same size as weights
def squared_loss_gradient(u, v, weights):

    print(u , v, weights)
    return dot(ktimesv(2, vector_subtraction(u, v)), weights)
    

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

        #perform forward pass and then calculate the loss 
        print("\nperforming front pass")
        pred = mlp(cur_X, weights, biases)

        loss = squared_loss(cur_y, pred)

        #for every layer in the network
        for layer in range(n_layers - 1, -1, -1):
            
            if(layer == (n_layers - 1)):
                print("output layer original weights: ", weights[layer])
                weights[layer] = vector_subtraction(weights[layer], ktimesv(learning_rate, squared_loss_gradient(pred, cur_y, weights[layer])))
                print("output layer updated weights: ", weights[layer])
            elif(layer == 0):
                print("input layer original weights: ", weights[layer])
                weights[layer] = vector_subtraction(weights[layer], ktimesv(learning_rate, squared_loss_gradient(pred, cur_y, weights[layer])))
                print("input layer updated weights: ", weights[layer])
            else:
                print("hidden layer original weights: ", weights[layer])
                weights[layer] = vector_subtraction(weights[layer], ktimesv(learning_rate, squared_loss_gradient(pred, cur_y, weights[layer])))
                print("hidden layer updated weights: ", weights[layer])




    return