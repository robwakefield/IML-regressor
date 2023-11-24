# Part 1



# Part 2

## Regressor

The Regressor model can be constructed with the arguments:
 - nb_epoch: The number of epochs to train the model
 - learning_rate: The initial learning rate to pass into the Adam GD model
 - hidden_layer_sizes: An array defining th shape of the hidden layers
 - batch_size: The batch size to use for mini-batch GD (default: -1 uses a single batch equal to the size of the dataset)
 - output_file: If True the per epoch RMSE will be written to loss.csv during training of the model

To train the model, pass training data when calling the fit method.
Pass test data to the predict method to obtain the models predicted values.
The score method will return the root mean squared error between the models predictions and the actual data.

## Hyperparameter Tuning

The param_grid in the RegressorHyperParameterSearch function defines which samples are to be used in the grid search to find the optimal parameters.

This function will return the best model's parameters and will also output all of the scores for each combination of hyperparameter to hyperparams_scores/all_hyperparams_scores.csv. This is useful for testing and validating the results.

Running the example_main() function will perform a hyperparameter search and then train a model based on the optimal parameters found - saving a pickle of it to part2_model.pickle

