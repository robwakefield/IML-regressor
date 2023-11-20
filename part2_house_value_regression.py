import torch
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from torch import nn

class Regressor():

    def __init__(self, x, nb_epoch = 1000, learning_rate=1e-6):
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
        self.hidden_size = 20
        self.nb_epoch = nb_epoch 
        self.learning_rate = learning_rate
        
        # Define NN structure
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
        )

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
        # TODO: Might be better to set to default value than 'forward-fill'
        x = x.fillna(method='ffill')

        # Convert ocean_proximity to separate columns
        lb = LabelBinarizer()
        lb.fit(x["ocean_proximity"])
        # TODO: Should we store this in class somewhere?
        ocean_classes = lb.classes_
        discretized = pd.DataFrame(lb.transform(x["ocean_proximity"]), columns=ocean_classes, dtype=np.float64)

        # Remove ocean_proximity from original DataFrame
        x = x.drop(["ocean_proximity"], axis=1)

        # Merge x and discretized ocean proximities
        merged = pd.merge(x, discretized, left_index=True, right_index=True)

        # Convert x to torch.tensor
        t_x = torch.from_numpy(merged.to_numpy()).to(torch.float32)
        
        # TODO: Normalise x

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
        
        # Define MSE loss func and Gradient Descent optimizer
        loss_fn = nn.MSELoss()
        optim = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

        for e in range(self.nb_epoch):
            # Perform forward pass though the model given the input.
            pred_Y = self.model(X)

            # Compute the loss based on this forward pass.
            loss = loss_fn(pred_Y, Y)
            if e < 10 or e % 100 == 0:
                print(f'Epoch {e}, Loss: {loss.item()}')

            # Perform backwards pass to compute gradients of loss with respect to parameters of the model.
            self.model.zero_grad()
            loss.backward()

            #TODO: torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Perform one step of gradient descent on the model parameters.
            optim.step()

            # You are free to implement any additional steps to improve learning (batch-learning, shuffling...).

        pred_Y = self.model(X)
        loss = loss_fn(pred_Y, Y)
        print("Expected:", Y)
        print("Actual:", pred_Y)
        print("Final Loss:", loss.item())

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
        pass

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
        return 0 # Replace this code with your own

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



def RegressorHyperParameterSearch(): 
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

    return  # Return the chosen hyper parameters

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
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 1000, learning_rate=0.001)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()

