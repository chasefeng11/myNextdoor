## 📚 Table of contents

- [Overview](#-overview)
- [Features] (#-features)
- [Disclaimers] (#-disclaimers)
- [Technical Components](#-technical-stack)
  - [Data Parsing](#-data-parsing)
  - [Data Analysis](#-data-analysis)
  - [Operations](#-operations)
- [Design](#-design)
- [Usage](#-usage)

## Overview

myNextdoor is a program which sends users daily emails containing recommended posts for them from their Nextdoor.com feed according to their individual preferences. It learns these preferences by scanning historical posts and and noting whether the user has interacted with them, which is then used as training data set for supervised learning models to predict whether a new post is of interest to the user based on text and quantitative factors.

## Features

-- [Parse](src/DataParser/) both daily and historical Nextdoor.com feed using Selenium Web Automation
-- [Preprocess](src/DataAnalysis.py) retrieved text and quantitative data to be made suitable for models
-- [Implement](src/) common supervised learning algorithms from scratch and train them on historical data to be able to generate predictions as to whether or not a new post is relevant to our user's interest
-- [Notify](src/main.py) our user new posts via automated SMTP emails and store these recommendations to a SQLite database 


## Disclaimers

myNextdoor is a research project and is not intended for any commercial use. The program can only access neighborhoods on Nextdoor.com that its user belongs to, and it is the user's job to treat all such restricted data with care and with respect to the privacy of those who belong in their neighborhood. No data has been published in this repository and all references to findings in this README.md file have been modified to respect the privacy of those beloging to the original creator's neighborhood. 


## 🛠 Technical stack
### Data Parsing
- Framework: [Selenium Web Driver](https://www.selenium.dev/documentation/webdriver/)
- Programming Language: [Java][https://www.java.com/en/]
### Data Analysis
- Programming Language: [Python](https://www.python.org/)
- Libraries: [sklearn](https://scikit-learn.org/), [numpy](https://numpy.org/), [pandas](https://pandas.pydata.org/)
## Operations
- Email Delivery: [SMTP](https://github.com/python/cpython/blob/3.11/Lib/smtplib.py)
- Database: [SQLite3](https://www.sqlite.org/index.html)
- Model Serialization: [Pickle](https://docs.python.org/3/library/pickle.html)
- Process Communication: [JSON](https://www.json.org/json-en.html), [subprocess](https://docs.python.org/3/library/subprocess.html)

