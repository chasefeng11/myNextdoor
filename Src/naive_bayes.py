import json
import math
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score

import DataAnalysis as da


class GaussianNaiveBayes:
    """
    Class for building and testing a Gaussian Naive Bayes model for text classification across two output classes.

    Assumptions
    1) Bag of Words: Within a post body, the location of keywords does not matter, only their frequency counts.
    2) Feature Independence: Our features (the keywords) occur independently of one another.
    3) Gaussian Distribution: Keyword frequencies follow a Gaussian distribution (no need for laplace smoothing 
                                                                                  since this is continous)
    """

    def __init__(self):
        # Prior probability, i.e. P(post = important)
        self.prior_probability = None

        # Likelihood probability, i.e. P(keyword | post = important) ~ Gaussian(mean, var)
        self.likelihood_means = None
        self.likelihood_vars = None


    def fit(self, X: np.array, y: np.array) -> None:
        """
        Train our model by calculating forms for our likelihood and posterior distributions. 

        param X: Feature matrix with their frequencies across all samples
        param y: Boolean truth labels of all samples
        """

        num_features, num_labels = X.shape

        # Compute our prior probability as the number of observed successes over the total number
        # of observations
        self.prior_probability = X[y == 1] / num_labels

        # Compute Gaussian mean and variance parameters for each feature/output pair
        self.likelihood_means = np.zeroes((2, num_features))
        self.likelihood_vars = np.zeroes((2, num_features))

        for output in [0, 1]:
            subarr = X[y == output]
            self.likelihood_means[output, :] = np.mean(subarr, axis=0)
            self.likelihood_vars[output, :] = np.var(subarr, axis=0)


    def predict(self, X: np.array):
        """
        Use the likelihood & posteriors to generate predictions for a new feature matrix X.

        param X: Feature matrix for a set of out-of-sample inputs
        """
        
        # If the posterior probability of true exceeds that of false, predict the post is
        # important
        predictions = np.array(np.zeroes(X.shape[0]))
        for i, row in enumerate(X):
            predictions[i] = int(self.calculate_label_probability(row, 1)
                                 >= self.calculate_label_probability(row, 0))
            
        return predictions


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

        return accuracy


    def calculate_label_probability(self, x: np.array, output: int):
        """
        Calculate the posterior probability that a row vector has the specified output.

        param x: Row vector corresponding to feature frequencies within a post
        param output: 1/0 to predict the posterior probability of true/false, respectively
        """
        
        # Note: We can simplify our math by using the log of our probabilities, and by ignoring the
        # denominator (which is the same for both posterior calculations)
        prior = math.log(self.prior_probability) if bool(output) else math.log(1 - self.prior_probability)

        gaussian_mean = self.likelihood_means[output]
        gaussian_var = self.likelihood_vars[output]

        posterior = np.sum(np.log(GaussianNaiveBayes.gaussian_pdf(x, gaussian_mean, gaussian_var))) + prior
        return posterior
        

    @staticmethod
    def gaussian_pdf(x: float, mean: float, var: float) -> float:
        """
        Evaluate the Gaussian pdf with the given mean and variance at specified x point.end=
        """
        return 1 / np.sqrt(var * 2 * np.pi) * np.exp(-1 / 2 * ((x - mean) ** 2 / var))
    

def frequency_matrix(text_samples: list) -> tuple:
    """
    Return frequency matrix of words appearing across a list of text samples along with the corresponding Vectorizer object.

    Example: [I am happy today!, We are happy today!] -> [1 0 1 1 0]
                                                         [0 1 1 1 1]
    """

    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(text_samples)


if '__name__' == '__main__':
    SCRIPT_PATH = __file__
    DATA_PATH = SCRIPT_PATH.replace('Src/explore.py', 'Data/data.json')

    with open(DATA_PATH) as fp:
        data = json.load(fp)
        full_df = da.preprocess_historical_dataset(data)

        # Get feature matrix containing frequencies of words in each post
        text = full_df['Text'].values()
        feature_matrix = frequency_matrix(text)

        # Get boolean labels (0 or 1) of each post
        labels = [int(val) for val in full_df['Interacted'].values()]
        labels = np.array(labels)

        # Train NB model with feature_matrix and labels
        model = GaussianNaiveBayes()
        model.fit(feature_matrix, labels)

        # Serialize model as pickle objects

        


        
        

