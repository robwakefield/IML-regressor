import torch
import pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV, train_test_split
from collections import OrderedDict
from torch import nn

class Regressor():

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Init weights using xavier
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                # Set bias to 0
                torch.nn.init.zeros_(m.bias)

    def __init__(self, x, nb_epoch = 1000, learning_rate=1, hidden_layers_sizes=[8], batch_size=-1, output_file=False):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        print("Using device ", self.device)
        X, _ = self._preprocessor(x, training = True)

        self.input_size = X.shape[1]
        self.output_size = 1
        self.hidden_layers_sizes = hidden_layers_sizes
        self.nb_epoch = nb_epoch 
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.output_file = output_file
        print("init Model Param:")
        print(f'epoch: {nb_epoch} learning rate: {learning_rate} hidden_layer: {hidden_layers_sizes} batch_size: {batch_size}')
        
        # Define NN structure
        layers = []
        layers.append((f'layer 0', nn.Linear(self.input_size, hidden_layers_sizes[0])))
        layers.append((f'Activation Function 0',nn.ReLU()))
        
        # Add no_of_layers hidden layer
        for i in range(0, len(hidden_layers_sizes) - 1):
            layers.append((f'layer {i + 1}', nn.Linear(self.hidden_layers_sizes[i], hidden_layers_sizes[i + 1])))
            layers.append((f'Activation Function {i + 1}',nn.ReLU()))

        layers.append((f'layer {len(hidden_layers_sizes)}', nn.Linear(hidden_layers_sizes[len(hidden_layers_sizes) - 1], self.output_size)))
        self.model = nn.Sequential(OrderedDict(layers))
        self.model.to(self.device)

        # Initialise weights using xavier and set bias to 0
        self.model.apply(self.init_weights)

        # Define MSE loss func and Gradient Descent optimizer
        self.loss_fn = nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        print("NN Model:")
        print(self.model)
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None
        
        # Fill Na values with default value
        x = x.fillna(value=0.0)

        # Convert ocean_proximity to separate columns
        ocean_classes = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
        lb = LabelBinarizer()
        lb.fit(ocean_classes)
        discretized = pd.DataFrame(lb.transform(x["ocean_proximity"]), columns=ocean_classes, dtype=np.float64)

        # Remove ocean_proximity from original DataFrame
        x = x.drop(["ocean_proximity"], axis=1)

        # Calculate min/max of each column for normalisation (only on training data)
        if training:
            self.max = x.max()
            self.min = x.min()
    
        # Min-Max normalise (columnwise)
        x = (x - self.min) / (self.max - self.min)

        # Merge x and discretized ocean proximities
        x = x.reset_index(drop=True)
        merged = x.join(discretized)
        merged = merged.astype('float32')

        # Convert x to torch.tensor
        t_x = torch.tensor(merged.to_numpy(), device=self.device)
        
        # Ensure data is correct shape
        assert t_x.shape[1] == 13

        if isinstance(y, pd.DataFrame):
            # Fill Na values in y
            # TODO: Is 0 the best value to fill here?
            y = y.fillna(value=0.0)
            y = y.astype('float32')
            
            # Convert y to torch.tensor
            t_y = torch.tensor(y.to_numpy(), device=self.device)
            t_y.to(self.device)

        return t_x, (t_y if isinstance(y, pd.DataFrame) else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
        
        batches = []
        if self.batch_size == -1:
            batches = [(X, Y)]
        else: 
            for i in range(0, X.size(0) - self.batch_size, self.batch_size):
                batches.append((X[i:i+self.batch_size], Y[i:i+self.batch_size]))

        
        print("Fit Model Param:")
        print(f'epoch: {self.nb_epoch} learning rate: {self.learning_rate} hidden_layer: {self.hidden_layers_sizes} batch_size: {self.batch_size}')

        losses = []
        
        for e in range(self.nb_epoch):
            # Perform forward pass though the model given the input.
            pred_Y = self.model(X)

            # Compute the loss based on this forward pass.
            loss = self.loss_fn(pred_Y, Y)
            losses.append(loss.item())        
            if self.nb_epoch <= 10 or e % 100 == 0:
                print(f'Epoch {e}, Loss: {loss.item()}')

            for X, Y in batches:
                # Perform forward pass though the model given the input.
                pred_Y = self.model(X)

                # Compute the loss based on this forward pass.
                loss = self.loss_fn(pred_Y, Y)

                # Perform backwards pass to compute gradients of loss with respect to parameters of the model.
                self.model.zero_grad()
                loss.backward()

                # Perform one step of gradient descent on the model parameters.
                self.optim.step()

        # Write losses to csv file
        if self.output_file:
            with open('loss.csv', 'w') as f:
                for i in range(len(losses)):
                    f.write(str(i)+','+str(losses[i])+'\n')

        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        return self.model(X).cpu().detach().numpy()

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget

        pred_Y = self.model(X)
        loss = self.loss_fn(pred_Y, Y)
        
        return torch.sqrt(loss).item()

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model

class RegressorAdaptor(BaseEstimator, RegressorMixin):
    def __init__(self, x_train, x_columns, y_columns, nb_epoch=1000, learning_rate=1000, hidden_layers_sizes=[5], batch_size=-1):
        self.x_train = x_train
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.hidden_layers_sizes = hidden_layers_sizes
        self.batch_size = batch_size
        self.x_columns = x_columns
        self.y_columns = y_columns
    
    # Change the inputs to DataFrame (To make compatible with previous code)
    def npArraytoDataFrame(self, data, columns):
        return pd.DataFrame(data, columns=columns)
        
    def fit(self, x, y):
        x = self.npArraytoDataFrame(x, self.x_columns)
        y = self.npArraytoDataFrame(y, self.y_columns)
        self.model = Regressor(x, self.nb_epoch, self.learning_rate, self.hidden_layers_sizes, self.batch_size)
        return self.model.fit(x, y)

    def predict(self, x):
        x = self.npArraytoDataFrame(x, self.x_columns)
        return self.model.predict(x)

    def score(self, x, y):
        x = self.npArraytoDataFrame(x, self.x_columns)
        y = self.npArraytoDataFrame(y, self.y_columns)
        return self.model.score(x, y)

def RegressorHyperParameterSearch(x_train, y_train): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    # Create scikit-learn estimator
    x_columns = x_train.columns
    y_columns = y_train.columns

    # We are using scikit learn estimator which uses numpy array
    x_train_np = x_train.to_numpy()
    y_train_np = y_train.to_numpy()

    regressor = RegressorAdaptor(x_train_np, x_columns, y_columns)


    # Define hyperparameter candidate we can optimise
    param_grid = {
        'learning_rate': [0.1, 1, 10],
        'hidden_layers_sizes': [[8], [16], [32], [64],
                                [16, 8], [32, 16], [64, 32],
                                [64, 32, 16]],
        'nb_epoch': [10, 100, 1000, 10000],
        'batch_size': [-1, 5000, 1000, 500]
    }

    grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

    grid_search.fit(x_train_np, y_train_np)

    print('Results of Cross Validation')

    params_combinations = grid_search.cv_results_['params']

    params_combinations_scores = [(params_combinations[i], grid_search.cv_results_['mean_test_score'][i]) for i in range(0, len(params_combinations))]

    printHyperParamsScores(param_grid, params_combinations_scores)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print('Best Model After Cross Validation')
    print('best_params:')
    print(best_params)
    print('best_score')
    print(best_score)
    return best_params

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################

def printHyperParamsScores(param_grid, params_combinations_scores):
    # Output all scores of all hyperparameters combinations
    params_combinations_scores_df = pd.DataFrame(params_combinations_scores, columns=['Params', 'Score'])
    params_combinations_scores_df.to_csv('hyperparams_scores/all_hyperparams_scores.csv', index=True)
    print(params_combinations_scores_df)

    # Output the mean performance aginst the change of one hyperparameter 
    for param in param_grid.keys():
        param_scores = []
        for param_candidate in param_grid[param]:
            mean_score = sum(score for (combination, score) in params_combinations_scores if (combination[param] == param_candidate))
            param_scores.append((param_candidate, mean_score))
        param_scores_df = pd.DataFrame(param_scores, columns=[param, 'Score'])
        param_scores_df.to_csv(f'hyperparams_scores/{param}_scores.csv')
        print(param_scores_df)

    # Output the mean performance aginst the change of two hyperparameters
    related_params = [('hidden_layers_sizes', 'nb_epoch'), ('hidden_layers_sizes', 'learning_rate')]
    for param_1, param_2 in related_params:
        param2_scores = []
        param_combinations = [(p1, p2) for p1 in param_grid[param_1] for p2 in param_grid[param_2]]
        for param_candidate_1, param_candidate_2 in param_combinations:
            mean_score = sum(score for (combination, score) in params_combinations_scores if (combination[param_1] == param_candidate_1) and (combination[param_2] == param_candidate_2))
            param2_scores.append((param_candidate_1, param_candidate_2, mean_score))
    param2_scores_df = pd.DataFrame(param2_scores, columns=[param_1, param_2, 'Score'])
    param2_scores_df.to_csv(f'hyperparams_scores/{param_1}_{param_2}_scores.csv')
    print(param2_scores_df)

def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    # Split training set and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 1000, learning_rate=1, hidden_layers_sizes=[64, 32], batch_size=500, output_file=True)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)
    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))

    # Find the best hyperparameters
    best_params = RegressorHyperParameterSearch(x_train, y_train)

    # The model with the best hyperparameters
    regressor = Regressor(x=x_train, **best_params, output_file=True)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()

