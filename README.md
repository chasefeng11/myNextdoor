## 📚 Table of contents

- [Overview](#-overview)
- [Features](#-features)
- [Disclaimers](#-disclaimers)
- [Technical Components](#-technical-stack)
  - [Data Parsing](#-data-parsing)
  - [Data Analysis](#-data-analysis)
  - [Operations](#-operations)
- [Design](#-design)

## 🔎 Overview

myNextdoor is a program which sends users daily emails containing recommended posts for them from their Nextdoor.com feed according to their individual preferences. It learns these preferences by scanning historical posts and and noting whether the user has interacted with them, which is then used as training data set for supervised learning models to predict whether a new post is of interest to the user based on text and quantitative factors.

## 📌 Features

- [Parse](src/DataParser/) both daily and historical Nextdoor.com feed using Selenium Web Automation
- [Preprocess](src/DataAnalysis.py) retrieved text and quantitative data to be made suitable for models
- [Implement](src/) common supervised learning algorithms from scratch and train them on historical data to be able to generate predictions as to whether or not a new post is relevant to our user's interest
- [Notify](src/main.py) our user new posts via automated SMTP emails and store these recommendations to a SQLite database 


## 📋 Disclaimers

myNextdoor is a research project and is not intended for any commercial use. The program can only access neighborhoods on Nextdoor.com that its user belongs to, and it is the user's job to treat all such restricted data with care and with respect to the privacy of those who belong in their neighborhood. No data has been published in this repository and all references to findings in this README.md file have been modified to respect the privacy of those beloging to the original creator's neighborhood. 


## 🛠 Technical Components
### Data Parsing
- Framework: [Selenium Web Driver](https://www.selenium.dev/documentation/webdriver/)
- Programming Language: [Java](https://www.java.com/en/)
### Data Analysis
- Programming Language: [Python](https://www.python.org/)
- Libraries: [sklearn](https://scikit-learn.org/), [numpy](https://numpy.org/), [pandas](https://pandas.pydata.org/)
### Operations
- Email Delivery: [SMTP](https://github.com/python/cpython/blob/3.11/Lib/smtplib.py)
- Database: [SQLite3](https://www.sqlite.org/index.html)
- Model Serialization: [Pickle](https://docs.python.org/3/library/pickle.html)
- Process Communication: [JSON](https://www.json.org/json-en.html), [subprocess](https://docs.python.org/3/library/subprocess.html)


## 🧮 Design

To generate insights, we abstract every Nextdoor post into the following fields
- Text body (i.e. "Isn't today a beautiful day!")
- Author (i.e. John Smith)
- Hometown of Author (i.e. Dallas, Fort Worth, etc.)
- How long ago it was posted (i.e. "1 hour ago", "7 days ago", etc.)
- Number of reactions it has received 
- Number of comments it has received 

We wish to analyze these qualitative/quantitative factors to return a prediction as to whether or not a post with some set of fields is either "important" or "non-important" to our user according to their individual preferences.

To learn these preferences, we must first compile some training data set of posts, labelled as either important or non-important. Ideally, this would come from the user themselves, but to approximate this, we implemented a [parser](Src/DataParser/HistoricalParser.java) to look back and parse historical posts and attach to each a boolean label capturing whether or not our user either 1) "reacted" to them (i.e. liked, supported, etc.) or 2) left a "comment" on them. In doing so, we assume that if our user interacts with a post of any of these ways, then he or she finds the post "important", and that we should recommended similar such posts to them in the future.

With this Java code, we build a labelled training dataset of ids, posts, and their "observed" importances. We save it as `.json` file, which might look something like this

```
  "285667624": {
    "Interacted": false,
    "Author": "Steven Smith",
    "NumReactions": 1,
    "Text": "Looking for beginner tennis lessons",
    "Age": "1 day ago",
    "NumComments": 0,
    "Location": "Neverland"
  },
  "285642325": {
    "Interacted": false,
    "Author": "Gilbert Miranda",
    "NumReactions": 33,
    "Text": "Does anyone know why the road off the highway is under construction?",
    "Age": "3 hours ago",
    "NumComments": 126,
    "Location": "Narnia"
  },
  "285658580": {
    "Interacted": true,
    "Author": "Madeleyne Lucero",
    "NumReactions": 1,
    "Text": "How should I train my dog?",
    "Age": "1 day ago",
    "NumComments": 1,
    "Location": "New York City"
  },
```

With this saved, we can then work on implementing some supervised learning algorithms to make predictions!

At a high level, we must generate insights from both our 1) text data and 2) our non-text data, which may be either qualitative (i.e. author) or quantitative (i.e. number of likes).
For analyzing text, I choose to implement a [Naive Bayes model](Src/naive_bayes.py), while for non-text data, I used [Logistic Regression model](Src/logistic_regression.py) after first converting all data points into quantitative measures. 

A [Naive Bayes Classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) relies on Bayes formula estimate the conditional probability of an outcome (i.e. whether a post is important) given particular observed features (i.e. a frequency vector of keywords that appear in that post). Mathematically, for a post with a vector of word frequencies $\vec{v}$, we compute the probability the post is important as 

$$
\begin{align}
P(\text{true} | \vec{v}) &= \frac{P(\text{true}) P(\vec{v} | \text{true})}{P(\vec{v})} \\
&\propto P(\text{true}) \prod P(f_i = v_i)
\end{align}
$$

where $P(\text{true})$ denotes the prior probability that any arbitrary post is important, and $P(f_i = v_i)$ indicates the feature likelihood probability that any arbitrary post contains the $i$-th word with $v_i$ frequency.

Using our training data set, we can calculate the values of these prior probabilities for each output class (i.e. important vs non-important), and the values of our feature likelihoods for each keyword ever observed. We assume each of our frequencies follow $Gaussian(\mu, \sigma^2)$ distribution, and we estimate each mean and variance parameter according to the training dataset. After this is complete, we now have a trained model, which we can use to generate boolean prediction on new posts by computing the posterior probabilities of each outcome, then choosing the outcome with the larger probability. 

Meanwhile, we use a [Logistic Regression model](https://en.wikipedia.org/wiki/Logistic_regression) to generate similar boolean predictions for each post, this time looking at the non-text parameters. This is similar to a Linear Regression algorithm, which computes an output level given a set of numerical inputs. However, a key distinction (which makes it suitable for this classification problem) is that a logistic regression "squishes" our output range to be between 0 and 1 for any input vector $\vec{x}$ according to a [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function), as seen below

$$
\sigma(\vec{x}) = \frac{1}{1 + e^{-\theta \cdot \vec{x} + \beta}}
$$


The goal of training our model is to compute the coefficient vector $\theta$ and the bias $\beta$ to maximize our model's predictive ability over the labelled training data set.

Two steps were required to achieve this. First, for the regression approach to make any sense, we had to devise a way to convert all of our unordered qualitative inputs to ordered quantitative inputs prior to training the model. We did this according to [one-hot encoding](https://en.wikipedia.org/wiki/One-hot), which adds features to our input set by replacing categorical levels with linearly independent unit vectors. Second, to compute the coefficient vector and bias, we used a simple [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) implementation to optimize these vectors iteratively. 

The scripts ``naive_bayes.py`` and ``logistic_regression.py`` train these models and serialize them as pickle files, which are then loaded by ``main.py`` when it runs daily and takes the union of these two models to generate out-of-sample predictions.  
 





