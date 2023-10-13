import math, random
from helpers import *
from datasets import *

#fully connected linear layer
def fc_linear_layer(x, w, b):

    #x * w
    output = dot(x, w)
    
    #add bias
    for i in range(len(dot(x, w))):

        output[i] = elementwise_add([output[i]], [b])[0]

    return output

#apply sigmoid function element wise to a vector
def sigmoid(vec):
    return [[(1/(1 + (math.exp(-1 * x)))) for x in tt] for tt in vec]

#single layer perceptron
def slp(x, w, b):
    return sigmoid(fc_linear_layer(x, w, b))

#multi-layer perceptron
def mlp(x, w_vec, b_vec):

    cache = []

    #make sure weights and biases vectors are same length
    if(len(w_vec) != len(b_vec)):
        print("Error in mlp() -> not the same amount of weights and biases!")
        exit(1)
    
    #start with input x value
    out = x

    #loop through weights and bias and apply nested single layer perceptron starting from the first
    for w, b in zip(w_vec, b_vec):
        out = slp(out, w, b)
        cache.append(out)

    return out, cache

#take the hyperparameters and initialize the weights
def initialize_weights(n_layers, hidden_units, input_dim, output_dim):

    weights = []

    for i in range(n_layers):

        #input to hidden weights
        if(i == 0):
            weights.append(generate_random_matrix(input_dim, hidden_units))
        
        #hidden to hidden layer weights
        elif((i > 0) and (i < (n_layers - 1))):
            weights.append(generate_random_matrix(hidden_units, hidden_units))

        #hidden to output weights
        else:
            weights.append(generate_random_matrix(hidden_units, output_dim))
    
    return weights

def initialize_biases(n_layers, hidden_units, output_dim):
    biases = []

    #for each layer
    for i in range(n_layers):
        
        #not output layer
        if(i < (n_layers - 1)):
            biases.append([random.uniform(-1, 1) for x in range(hidden_units)])

        #output layer
        else:
            biases.append([random.uniform(-1, 1) for x in range(output_dim)])

    return biases

#loss function
def loss_fn(y, pred):
    
    # (u - v) * (u - v)
    u_minus_v = elementwise_sub(y, pred)
    return dot(u_minus_v, u_minus_v)

#cost function
def cost(y, pred):
    return elementwise_sub(y, pred)

#sigmoid derivative of a value thats already been sigmoided
def sigmoid_derivative(sig):
    one_minus_sig = [[(1-element) for element in row] for row in sig]
    return dot(sig, one_minus_sig)

#train MLP
def train_and_test(dataset, hyperparameters, learning_rate, num_epochs, train_test_split):

    #create and partition dataset
    X, y = generate_dataset(dataset)
    X_train, y_train, X_test, y_test = train_test_slit(X, y, train_test_split)

    #set up parameters
    n_hid_layers = hyperparameters[0] #number of hidden layers
    n_layers = n_hid_layers + 1
    input_dim = len(X[0]) #input dimensions
    output_dim = len(y[0]) #output dimensions
    hidden_units = hyperparameters[1] #hidden units in each layer

    #initialize weights and biases
    weights = initialize_weights(n_layers, hidden_units, input_dim, output_dim)
    biases = initialize_biases(n_layers, hidden_units, output_dim)

    #for the number of pass throughs we want of the data
    for epoch in range(1, num_epochs + 1):
        
        #front pass
        out, cache = mlp(X_train, weights, biases)
        
        prev_grad = []

        #go backward through the layers
        for layer in range(n_layers - 1, -1, -1):

            #hidden to output layer
            if(layer == (n_layers - 1)):

                #update weights
                further_output = ktimesv(2, elementwise_sub(y_train, out)) 
                deriv_from_front_pass = sigmoid_derivative(cache[layer])
                output_grad_without_input_to_layer = dot(further_output, deriv_from_front_pass)
                input_to_layer = cache[layer - 1]

                weights[layer] = elementwise_sub(weights[layer], ktimesv(learning_rate, dot(transpose(input_to_layer), output_grad_without_input_to_layer)))

                #update biases
                for row in ktimesv(learning_rate, output_grad_without_input_to_layer):
                    biases[layer] = elementwise_sub([biases[layer]], [row])[0]

                #save gradient for next layer
                prev_grad = output_grad_without_input_to_layer

            #hidden to hidden layer
            elif((layer != (n_layers - 1)) and (layer != 0)):
                
                #update weights
                further_output = dot(prev_grad, transpose(weights[layer + 1]))
                deriv_from_front_pass = sigmoid_derivative(cache[layer])
                hid_grad_without_input_to_layer = dot(further_output, deriv_from_front_pass)
                input_to_layer = cache[layer - 1]
                weights[layer] = elementwise_sub(weights[layer], ktimesv(learning_rate, dot(transpose(input_to_layer), hid_grad_without_input_to_layer))) 

                #update biases
                for row in ktimesv(learning_rate, hid_grad_without_input_to_layer):
                    biases[layer] = elementwise_sub([biases[layer]], [row])[0]

                #save gradient for next layer
                prev_grad = hid_grad_without_input_to_layer

            #input to hidden layer
            else:

                #update weights
                further_output = dot(prev_grad, transpose(weights[layer + 1]))
                deriv_from_front_pass = sigmoid_derivative(cache[layer])
                input_grad_without_input_to_layer = dot(further_output, deriv_from_front_pass)
                input_to_layer = X_train
                weights[layer] = elementwise_sub(weights[layer], ktimesv(learning_rate, dot(transpose(input_to_layer), input_grad_without_input_to_layer))) 

                #update biases
                for row in ktimesv(learning_rate, input_grad_without_input_to_layer):
                    biases[layer] = elementwise_sub([biases[layer]], [row])[0]

    test_losses = []

    #go through test data
    for x, y in zip(X_test, y_test):
        in_x = [x]

        #make a prediction
        pred, cache = mlp(in_x, weights, biases)

        #save the loss from that data point
        test_losses.append(loss_fn([y], pred)[0][0])

    #return the average of all the data point losses (which is the test loss)
    return sum(test_losses) / len(test_losses)