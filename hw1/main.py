from nn import *
from datasets import *

def main():

    #should be able to try changing combinations of the following:
        #number of layers
        #number of units in each hidden layer
        #learning rate
        #train test split
        #epochs
    splits_to_try = [0.5, 0.75, 0.9]
    hyperparameters_to_try = [(1, 5), (1, 10), (2, 5), (2, 10), (3, 5)] #these are in the form (number of hidden layers, number of units per layer)
    learning_rates_to_try = [0.001, 0.01, 0.1]
    epochs_to_try = [100, 500, 1000, 5000]

    xor_best_loss = 1.0
    xor_best_hyperparameters = tuple()
    xor_best_epochs = 0.0
    xor_best_split = 0.0
    xor_best_lr = 0.0

    for split in splits_to_try:
            for hyperparameters in hyperparameters_to_try:
                for lr in learning_rates_to_try:
                    for epochs in epochs_to_try:
                        loss = train2('xor', hyperparameters, lr, epochs, split)
                        print_train_status('xor', hyperparameters[0], hyperparameters[1], lr, epochs, split, loss)
                        if(loss < xor_best_loss):
                            xor_best_loss = loss
                            xor_best_hyperparameters = hyperparameters
                            xor_best_lr = lr
                            xor_best_epochs = epochs
                            xor_best_split = split

    adder_best_loss = 1.0
    adder_best_hyperparameters = tuple()
    adder_best_epochs = 0.0
    adder_best_split = 0.0
    adder_best_lr = 0.0

    for split in splits_to_try:
        for hyperparameters in hyperparameters_to_try:
            for lr in learning_rates_to_try:
                for epochs in epochs_to_try:
                    loss = train2('adder', hyperparameters, lr, epochs, split)
                    print_train_status('adder', hyperparameters[0], hyperparameters[1], lr, epochs, split, loss)
                    if(loss < adder_best_loss):
                        adder_best_loss = loss
                        adder_best_hyperparameters = hyperparameters
                        adder_best_lr = lr
                        adder_best_epochs = epochs
                        adder_best_split = split

    print("\nThe best network I tested for the xor dataset has the following details:")
    print("hidden layers:", xor_best_hyperparameters[0])
    print("units per hidden layer:", xor_best_hyperparameters[1])
    print("learning rate:", xor_best_lr)
    print("epochs:", xor_best_epochs)
    print("train/test split:", xor_best_split)
    print("testloss:", xor_best_loss, '\n')
    print("------------------------------------------------------------------------------------------------------")

    print("\nThe best network I tested for the adder dataset has the following details:")
    print("hidden layers:", adder_best_hyperparameters[0])
    print("units per hidden layer:", adder_best_hyperparameters[1])
    print("learning rate:", adder_best_lr)
    print("epochs:", adder_best_epochs)
    print("train/test split:", adder_best_split)
    print("testloss:", adder_best_loss, '\n')
    print("------------------------------------------------------------------------------------------------------")

    '''for dataset in datasets_to_try:

        best_overall_loss = 1.0

        best_loss = 1.0
        best_hyperparameters = (0,0)
        for hyperparameters in hyperparameters_to_try:
            loss = train2(dataset, hyperparameters, 0.01, 1000, 0.8)
            print_train_status(dataset, hyperparameters[0], hyperparameters[1], 0.01, 1000, 0.8, loss)
            if(loss < best_loss):
                best_loss = loss
                best_hyperparameters = hyperparameters
            if(loss < best_overall_loss):
                best_overall_loss = loss
                best_overall_hyperparameters = hyperparameters
                best_overall_lr = 0.01
                best_overall_epochs = 1000
                best_overall_split = 0.8

        print("best loss for hyperparameters test was:", best_loss, "with the parameters", best_hyperparameters)
        
        best_loss = 1.0
        best_lr = 0.0
        for lr in learning_rates_to_try:
            loss = train2(dataset, best_hyperparameters, lr, 1000, 0.8)
            print_train_status(dataset, best_hyperparameters[0], best_hyperparameters[1], lr, 1000, 0.8, loss)
            if(loss < best_loss):
                best_loss = loss
                best_lr = lr
            if(loss < best_overall_loss):
                best_overall_loss = loss
                best_overall_hyperparameters = best_hyperparameters
                best_overall_lr = lr
                best_overall_epochs = 1000
                best_overall_split = 0.8

        print("best loss for learning rate test was:", best_loss, "with a learning rate of", best_lr)

        best_loss = 1.0
        best_epochs = 0.0
        for epochs in epochs_to_try:
            loss = train2(dataset, best_hyperparameters, best_lr, epochs, 0.8)
            print_train_status(dataset, best_hyperparameters[0], best_hyperparameters[1], best_lr, epochs, 0.8, loss)
            if(loss < best_loss):
                best_loss = loss
                best_epochs = epochs
            if(loss < best_overall_loss):
                best_overall_loss = loss
                best_overall_hyperparameters = best_hyperparameters
                best_overall_lr = best_lr
                best_overall_epochs = epochs
                best_overall_split = 0.8

        print("best loss for epochs test was:", best_loss, "with", best_epochs, "epochs")

        best_loss = 1.0
        best_split = 0.0
        for split in splits_to_try:
            loss = train2(dataset, best_hyperparameters, best_lr, best_epochs, split)
            print_train_status(dataset, best_hyperparameters[0], best_hyperparameters[1], best_lr, best_epochs, split, loss)
            if(loss < best_loss):
                best_loss = loss
                best_split = split
            if(loss < best_overall_loss):
                best_overall_loss = loss
                best_overall_hyperparameters = best_hyperparameters
                best_overall_lr = best_lr
                best_overall_epochs = epochs
                best_overall_split = split'''

        #print("best loss for training split test was:", best_loss, "with training split", best_split)
        
    '''print("\nSo, the best network I tested for the", dataset, "dataset has the following details:")
    print("hidden layers:", best_hyperparameters[0])
    print("units per hidden layer:", best_hyperparameters[1])
    print("learning rate:", best_lr)
    print("epochs:", best_epochs)
    print("train/test split:", best_split)
    print("loss:", best_loss, '\n')
    print("------------------------------------------------------------------------------------------------------")'''


if __name__ == "__main__":
    main()