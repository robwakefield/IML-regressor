import torch
import pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array
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

    def __init__(self, x, nb_epoch = 1000, learning_rate=1e-6, hidden_layers_sizes=[5]):
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
        X, _ = self._preprocessor(x, training = True)

        self.input_size = X.shape[1]
        self.output_size = 1
        self.hidden_layers_sizes = hidden_layers_sizes
        self.nb_epoch = nb_epoch 
        self.learning_rate = learning_rate
        print("init Model Param:")
        print(f'epoch: {nb_epoch} learning rate: {learning_rate} hidden_layer: {hidden_layers_sizes}')
        
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

        # Convert x to torch.tensor
        t_x = torch.from_numpy(merged.to_numpy()).to(torch.float32)
        
        # Ensure data is correct shape
        assert t_x.shape[1] == 13

        if isinstance(y, pd.DataFrame):
            # Fill Na values in y
            # TODO: Is 0 the best value to fill here?
            y = y.fillna(value=0.0)
            
            # Convert y to torch.tensor
            t_y = torch.from_numpy(y.to_numpy()).to(torch.float32)

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
        print("Fit Model Param:")
        print(f'epoch: {self.nb_epoch} learning rate: {self.learning_rate} hidden_layer: {self.hidden_layers_sizes}')
        
        for e in range(self.nb_epoch):
            # Perform forward pass though the model given the input.
            pred_Y = self.model(X)

            # Compute the loss based on this forward pass.
            loss = self.loss_fn(pred_Y, Y)
            if self.nb_epoch <= 10 or e % 100 == 0:
                print(f'Epoch {e}, Loss: {loss.item()}')

            # Perform backwards pass to compute gradients of loss with respect to parameters of the model.
            self.model.zero_grad()
            loss.backward()

            #TODO: torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Perform one step of gradient descent on the model parameters.
            self.optim.step()

            # You are free to implement any additional steps to improve learning (batch-learning, shuffling...).

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
        return self.model(X).detach().numpy()

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
    def __init__(self, x_train, x_columns, y_columns, nb_epoch=1000, learning_rate=1000, hidden_layers_sizes=[5]):
        self.x_train = x_train
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.hidden_layers_sizes = hidden_layers_sizes
        self.x_columns = x_columns
        self.y_columns = y_columns
    
    # Change the inputs to DataFrame (To make compatible with previous code)
    def npArraytoDataFrame(self, data, columns):
        return pd.DataFrame(data, columns=columns)
        
    def fit(self, x, y):
        x = self.npArraytoDataFrame(x, self.x_columns)
        y = self.npArraytoDataFrame(y, self.y_columns)
        self.model = Regressor(x, self.nb_epoch, self.learning_rate, self.hidden_layers_sizes)
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

    # Define hyperparameter we can optimise
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1, 1, 10],
        'hidden_layers_sizes': [[8], [16], [32], [64],
                                [16, 8], [32, 16], [64, 32],
                                [64, 32, 16]],
        # 'learning_rate': [10],
        # 'hidden_layers_sizes': [[7], [13]],
    }

    # check_estimator(regressor)

    grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

    grid_search.fit(x_train_np, y_train_np)

    print('Results of Cross Validation')
    print([(grid_search.cv_results_['params'][i], grid_search.cv_results_['mean_test_score'][i])
            for i in range(0, len(grid_search.cv_results_['params']))])
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print('Best Model After Cross Validation')
    print('best_params:')
    print(best_params)
    
    best_score = best_model.score(x_train_np, y_train_np)
    print("Best RMSE:", best_score)

    return best_params

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



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
    regressor = Regressor(x_train, nb_epoch = 1000, learning_rate=10)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)
    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))

    # Find the best hyperparameters
    best_params = RegressorHyperParameterSearch(x_train, y_train)

    # The model with the best hyperparameters
    regressor = Regressor(x=x_train, **best_params)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()

