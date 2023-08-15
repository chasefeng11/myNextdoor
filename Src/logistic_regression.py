import json
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import DataAnalysis as da

class LogisticRegression:
    """
    Class for building and testing a Logistic Regression model for text classification across two output classes.

    Assumptions
    1) Independence of Errors
    2) Linearity in the Logit Function for Each Variable
    2) No Strongly Influential Outliers
    3) No Multicollinearity (one-hot encoding does not introduce multicollinearity because these vectors are
                             linearly independent)
    """

    def __init__(self, num_iterations: int = 5000, learning_rate: float = 0.001):
        """
        param learning_rate: Speed to adjust coefficients in gradient descent
        param num_iterations: Number of iterations to carry out gradient descent
        """

        self.n_iterations = num_iterations
        self.learning_rate = learning_rate

        # Vector of regression coefficients in sigmoid function
        self.coeffs = None

        # Value of bias in sigmoid function
        self.bias = 0


    def fit(self, X: np.array, y: np.array):
        """
        Pick regression coefficients with maximum likelihood using gradient descent.

        param X: Matrix of training inputs
        param y: Boolean truth labels of training inputs
        """

        num_labels, num_features = X.shape

        self.coeffs = np.zeros(num_features)

        for i in range(self.n_iterations):
            # Generate y prediction based on the log of the standard linear model
            y_hat = self.sigmoid(np.dot(X, self.coeffs) + self.bias)

            # Compute our gradient vectors and update our parameters
            coeffs_grad = (1 / num_labels) * np.dot(X.T, (y_hat - y))
            bias_grad = (1 / num_labels) * np.sum(y_hat - y)

            self.coeffs -= self.learning_rate * coeffs_grad 
            self.bias -= self.learning_rate * bias_grad


    def predict(self, X: np.array) -> np.array:
        """
        Use the regression coefficients generate predictions for a new matrix of samples X.

        param X: Matrix of testing inputs
        """
        y_hat = LogisticRegression.sigmoid(np.dot(X, self.coeffs) + self.bias)
        return np.array([1 if i > 0.5 else 0 for i in y_hat])
    

    def test(self, X: np.array, y: np.array):
        """
        Return our trained model's accuracy against a set of labelled data.
        """

        # Compute the model's prediction vector
        y_hat = self.predict(X)

        # Count the number of true positives, true negatives, false positives and false negatives
        num_items = X.shape[0]
        true_pos = ((y_hat == 1) & (y == 1)).sum()
        true_negs = ((y_hat == 0) & (y == 0)).sum()
        false_pos = ((y_hat == 1) & (y == 0)).sum()
        false_negs = num_items - true_pos - true_negs - false_pos

        # Print the test's overall accuracy, alpha, beta, and power
        accuracy = (true_pos + true_negs) / num_items
        alpha = false_negs / (false_negs + true_pos)
        beta = false_pos / (false_pos + true_negs)

        print("Test results for {} data points...".format(num_items))
        print("Accuracy: {}%".format(round(100 * accuracy, 2)))
        print("Alpha Error: {}%".format(round(100 * alpha, 2)))
        print("Beta Error: {}%".format(round(100 * beta, 2)))
        print("Power: {}%".format(round(100 * (1 - beta), 2)))

        print(classification_report(y, y_hat, target_names=['Non-Important', 'Important']))

        return accuracy
    
    @property
    def coefficients(self) -> list:
        return self.coeffs.tolist()

    @staticmethod
    def sigmoid(x: float):
        return 1 / (1 + np.exp(-x))
    

def one_hot_encode(non_binary_values: pd.Series) -> pd.DataFrame:
    """
    Convert a series of categorial data points to numerical vectors.
    """

    vals = non_binary_values.unique()
    indexes = {val: i for i, val in enumerate(vals)}
    
    data = []
    for index, value in non_binary_values.items():
        row = [0] * len(vals)
        row[indexes[value]] = 1
        data.append(row)

    return pd.DataFrame(data, columns=vals)


if __name__ == '__main__':
    SCRIPT_PATH = __file__
    DATA_PATH = SCRIPT_PATH.replace('Src/logistic_regression.py', 'Data/data.json')
    MODEL_PATH = SCRIPT_PATH.replace('Src/logistic_regression.py', 'Models/logistic_regression.pickle')

    with open(DATA_PATH) as input_fp:
        data = json.load(input_fp)
        full_df = da.preprocess_historical_dataset(data)

        # Get input matrix after encoding categorical inputs as row vectors
        count_df = full_df[['NumReactions', 'NumComments', 'Age']]
        author_df = one_hot_encode(full_df['Author']).set_index(count_df.index)
        location_df = one_hot_encode(full_df['Location']).set_index(count_df.index)
        target_df = pd.concat([count_df, author_df, location_df], axis=1)

        # Get boolean labels (0 or 1) of each post
        labels = [int(val) for val in full_df['Interacted'].to_list()]
        labels = np.array(labels)

        # Split historical dataset into training and test subsets
        X_train, X_test, y_train, y_test = train_test_split(target_df.to_numpy(), labels, test_size=0.2)

        # Train Logistic Regression model with training matrix and labels
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Print input variables and their corresponding columns
        print("Regression Coefficients...")
        for col, coeff in zip(target_df.columns, model.coefficients):
            print("{}, {}".format(col, coeff))

        # Test Logistic Regression on out-of-sample data
        model.test(X_test, y_test)

        # Serialize model as pickle object
        with open(MODEL_PATH, "wb") as output_fp:
            pickle.dump(model, output_fp) 
            print("Serialized Logistic Regression Model as a pickle file in {}".format(MODEL_PATH))
