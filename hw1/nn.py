import math, random
from helpers import *
import matplotlib.pyplot as plt
import numpy as np

#multiple input and weights, then add bias
def fc_linear_layer(x, w, b):
    return dot(x, w)
    #return vector_add(dot(w, x), b)

#sigmoid function element wise to a vector
def sigmoid(vec):
    #return 1 / (1 + np.exp(-vec))
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

        #this is the input layer weights
        if(i == 0):
            weights.append(generate_random_matrix(input_dim, hidden_units))
        
        #this is a middle hidden layer (i.e., not the output layer)
        elif((i > 0) and (i < (n_layers - 1))):
            weights.append(generate_random_matrix(hidden_units, hidden_units))

        #this is the output layer
        else:
            weights.append(generate_random_matrix(hidden_units, output_dim))
    
    return weights

def initialize_biases(n_layers, hidden_units, output_dim):
    biases = []

    #for each layer
    for i in range(n_layers):
        
        #create bias vector with same size as output of that layer
        #this is the input layer weights
        if(i < (n_layers - 1)):
            biases.append([random.uniform(-1, 1) for x in range(hidden_units)])

        #this is the output layer
        else:
            biases.append([random.uniform(-1, 1) for x in range(output_dim)])
    #print("bioas:", biases)
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

#loss function
def loss_fn(y, pred):
    
    # (u - v) * (u - v)
    u_minus_v = elementwise_sub(y, pred)
    return dot(u_minus_v, u_minus_v)

#loss function
def cost(y, pred):
    
    # (u - v) * (u - v)
    return elementwise_sub(y, pred)

def sigmoid_derivative(sig):
    one_minus_sig = [[(1-element) for element in row] for row in sig]
    return dot(sig, one_minus_sig)

#this must return something that is the same size as weights
def gradient(sigmoid_deriv, wildcard, input_to_layer, mode):
    
    if(mode == "weights"):
        output = dot(sigmoid_deriv, dot(wildcard, input_to_layer))

    else:
        output = dot(sigmoid_deriv, wildcard)

    return output

def step(val):
    if(val >= 1.0):
        return 1
    return 0

#train the network
def train(X_train, y_train, X_test, y_test, n_layers, input_dim, output_dim, hidden_units, learning_rate, epochs):

    #log the details of this training
    #print_train_status(X, y, n_layers, input_dim, output_dim, hidden_units, learning_rate, train_test_split)

    weights = initialize_weights(n_layers, hidden_units, input_dim, output_dim)
    
    biases = initialize_biases(n_layers, hidden_units, output_dim)

    print(weights[0])
    print(weights[1])

    for epoch in range(1, epochs + 1):
        print("epoch", epoch)
        out, cache = mlp(X_train, weights, biases)
        layer1 = cache[0]
        layer2 = cache[1]
    
        layer2_error = cost(out, y_train)
        
        layer2_delta = dot(layer2_error, sigmoid_derivative(layer2))

        layer1_error = dot(layer2_delta, transpose(weights[1]))
        layer1_delta = dot(layer1_error, sigmoid_derivative(layer1))

        weights[1] = elementwise_sub(weights[1], ktimesv(learning_rate, dot(transpose(layer1), layer2_delta)))
        weights[0] = elementwise_sub(weights[0], ktimesv(learning_rate, dot(transpose(X_train), layer1_delta)))
        
        
    print("final weights")
    print(weights[0])
    print(weights[1])    

    accurate = 0
    losses = []
    predictions = []

    for x, y in zip(X_test, y_test):
        in_x = [x]
        pred, cache = mlp(in_x, weights, biases)
        losses.append(loss_fn([y], pred)[0][0])
        predictions.append(pred[0][0])
        if(int(pred[0][0] > 0.5) == y[0]):
            accurate += 1

    print("final accuracy: ", float(accurate) / float(len(X_test)))
    print("final test loss:", sum(losses) / len(predictions))
    

    '''for epoch in range(1, epochs + 1, 1):

        #losses and num correct predictions for this epoch
        loss_total = 0.0

        #print(weights)

        #for each training X (i.e, an epoch)
        for cur_X, cur_y in zip(X_train, y_train):

            #perform forward pass and add prediction to predictions vector
            #print("forward passing")
            pred, cache = mlp(cur_X, weights, biases)
            #print("pred:", pred, "expected:", cur_y)
            #print("cached outputs:", cache)
            #print("loss:", squared_loss(cur_y, pred)[0][0])

            #compute loss and append it to the losses array
            loss_total += loss(cur_y, pred)[0][0]
            

            #print("performing back pass")

            #BACKPROPAGATION
            prev_grads = tuple()

            #for every layer in the network
            for layer in range(n_layers - 1, -1, -1):

                #hidden to output layer
                if(layer == (n_layers - 1)):

                    #print("\nold values for weights and biases in output layer:", weights[layer], biases[layer])

                    #compute bias and weight gradients, then save them for next layer
                    weight_grad = gradient(sigmoid_derivative(cache[layer]), ktimesv(2, vector_subtraction(cur_y, pred)), transpose(cache[layer - 1]), "weights")
                    bias_grad = gradient(sigmoid_derivative(cache[layer]), ktimesv(2, vector_subtraction(cur_y, pred)), None, "bias")
                    prev_grads = (weight_grad, bias_grad)

                    #print("gradients computed for output layer              : ", prev_grads[0], prev_grads[1])

                    #update weights and bias
                    weights[layer] = vector_subtraction(weights[layer], (ktimesv(learning_rate, weight_grad)))
                    biases[layer] = vector_subtraction(biases[layer], (ktimesv(learning_rate, bias_grad)))
                    
                    #print("new values for weights and biases in output layer:", weights[layer], biases[layer])

                    
                
                #input layer
                elif(layer == 0):

                    #print("\nold values for weights and biases in output layer:", weights[layer], biases[layer])

                    #compute bias and weight gradients, then save them for next layer
                    weight_grad = gradient(sigmoid_derivative(cache[layer]), dot(transpose(weights[layer + 1]), prev_grads[1]), transpose(cur_X), "weights")
                    bias_grad = gradient(sigmoid_derivative(cache[layer]), dot(transpose(weights[layer + 1]), prev_grads[1]), None, "bias")
                    prev_grads = (weight_grad, bias_grad)

                    #print("gradients computed for output layer              : ", prev_grads[0], prev_grads[1])

                    #update weights and bias
                    weights[layer] = vector_subtraction(weights[layer], (ktimesv(learning_rate, weight_grad)))
                    biases[layer] = vector_subtraction(biases[layer], (ktimesv(learning_rate, bias_grad)))
                    
                    #print("new values for weights and biases in output layer:", weights[layer], biases[layer])

                    #exit(0)

                #hidden to hidden layer
                else:

                    #print("\nold values for weights and biases in output layer:", weights[layer], biases[layer])

                    #compute bias and weight gradients, then save them for next layer
                    weight_grad = gradient(sigmoid_derivative(cache[layer]), dot(transpose(weights[layer + 1]), prev_grads[1]), transpose(cache[layer - 1]), "weights")
                    bias_grad = gradient(sigmoid_derivative(cache[layer]), dot(transpose(weights[layer + 1]), prev_grads[1]), None, "bias")
                    prev_grads = (weight_grad, bias_grad)

                    #print("gradients computed for output layer              : ", prev_grads[0], prev_grads[1])

                    #update weights and bias
                    weights[layer] = vector_subtraction(weights[layer], (ktimesv(learning_rate, weight_grad)))
                    biases[layer] = vector_subtraction(biases[layer], (ktimesv(learning_rate, bias_grad)))
                    
                    #print("new values for weights and biases in output layer:", weights[layer], biases[layer])

                    #exit(0)

        print("epoch", epoch, "loss:", (loss_total / len(X_train)))
        losses.append((loss_total / len(X_train)))

        #update best loss and accuracy for model if this epoch bested them
        if((loss_total / len(X_train)) < best_loss):
            best_loss = (loss_total / len(X_train))

    #plt.plot(range(0, epochs, 1), losses)
    #plt.show()

    print('\nAfter training:')
    #print("Best training accuracy:", best_accuracy)
    print("Best loss:", best_loss)'''

    return