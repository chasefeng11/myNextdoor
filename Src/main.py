import json
import pickle
import nltk
import pandas as pd
import subprocess
from subprocess import Popen
import smtplib, ssl
import DataAnalysis as dm
import sqlite3
import datetime
import sys
import numpy as np

import DataAnalysis as da
import utils
from utils import NextdoorDBConnection


# Path to referenced files
SCRIPT_PATH = __file__
CONFIG_PATH = SCRIPT_PATH.replace('Src/main.py', 'settings.config')
DB_PATH = SCRIPT_PATH.replace('Src/main.py', 'Databases/nextdoor.db')
RUNNER_PATH = SCRIPT_PATH.replace('Src/main.py', 'DataParser/src/Runner.java')

# Path to referenced directories
LOGS_PATH = SCRIPT_PATH.replace('Src/main.py', 'Logs/')
MODELS_PATH = SCRIPT_PATH.replace('Src/main.py', 'Models/')

# Database table to read/write from
DB_TABLE = 'postRecords'

# Runs mainScraper.pl from command line, reads STDOUT, and converts received JSON
# to a Python dictionary.
# Prints error message and quits if mainScraper passes a) an output than cannot
# be converted into a dict, or b) an empty output.

if __name__ == "__main__":
    date = datetime.date.today()

    javacmd = "java {}".format(RUNNER_PATH)
    subprocess = subprocess.Popen(javacmd, shell=True, stdout=subprocess.PIPE)
    data = subprocess.stdout.read()

    try:
        daily_posts = json.loads(data)
        if len(daily_posts) == 0:
            msg = "No posts were found! Problem with scraping?"
            utils.append_to_log(date, msg, LOGS_PATH + "nextdoor.err")
            sys.exit(msg)
    except json.decoder.JSONDecodeError:
        msg = "Scraping output could not be interpreted! Problem with scraping?"
        utils.append_to_log(date, msg, LOGS_PATH + "nextdoor.err")
        sys.exit(msg)

    # Instantiate database connection object
    nextdoor_db = NextdoorDBConnection(DB_PATH, DB_TABLE)

    # Load classifier & regression models
    bayes_model = pickle.load(open(MODELS_PATH + 'naive_bayes.pickle', "rb"))
    regression_model = pickle.load(open(MODELS_PATH + 'logistic_regression', "rb"))

    # Preprocess input data and generate respective input matrices for each model
    daily_posts = da.preprocess_dataset(daily_posts)
    words_matrix = da.word_frequency_matrix(daily_posts)
    num_matrix = da.numerical_factor_matrix(daily_posts).to_numpy()

    # Generate predictions from each model and take the union of the two for our overall prediction
    bayes_predictions = bayes_model.predict(words_matrix)
    regression_predictions = regression_model.predict(num_matrix)
    outputs = pd.Series(np.add(bayes_predictions, regression_predictions))
    daily_posts['Prediction'] = outputs

    # Categorize today's important posts as either 1) newly-recommended or 2) previously-recommended
    important_posts = daily_posts.loc[daily_posts['Prediction'] == 1]
    prev_ids = set(nextdoor_db.get_prior_ids())
    new_ids, new_urls, new_titles, new_authors = [], [], [], []
    old_ids, old_urls, old_titles, old_authors = [], [], [], []

    for id, row in important_posts.iterrows():
        url = utils.get_link_from_id(id)
        title = important_posts.iloc[id]['Title']
        author = important_posts.iloc[id]['Author']

        if id not in prev_ids:
            new_ids.append(id)
            new_urls.append(url)
            new_titles.append(title)
            new_authors.append(author)
            nextdoor_db.add_row_to_db(id, author)
        else:
            old_ids.append(id)
            old_urls.append(url)
            old_titles.append(title)
            old_authors.append(author)
            nextdoor_db.update_row_in_db(id, date)

    # If a post was previously recommended but was not recommended today, reset its streak in the database
    outdated_ids = [id for id in prev_ids if id not in old_ids]
    for id in outdated_ids:
        nextdoor_db.reset_streak(id)


    # Write and send email
    # Format: 1) Introduction, 2) New Recommendations, 3) Repeated Recommendations
    smtp_server, port, sender_email, sender_passwrd, receiver_emails = utils.get_smtp_settings(CONFIG_PATH)

    message = """\
    Subject: Daily Message from the myNextdoor

    Hello,

    Here are today's important posts. Click the links to see the post on Nextdoor.com


    New Posts:

    """

    for id, author, url, title in zip(new_ids, new_authors, new_urls, new_titles):
        message += "{} \n".format(title)
        message += "By: {} \n".format(author)
        message += "{} \n".format(url)
        message += "\n"

    message += "\n"
    message += "Posts that have been previously recommended:"
    message += "\n"
    message += "\n"

    for id, author, url, title in zip(old_ids, old_authors, old_urls, old_titles):
        message += "{} \n".format(title)
        message += "By: {} \n".format(author)
        message += "{} \n".format(url)
        message += "\n"


    # Create a secure SSL context
    context = ssl.create_default_context()

    # Try to log in to server and send email
    try:
        server = smtplib.SMTP(smtp_server,port)
        server.ehlo()
        server.starttls(context=context)
        server.ehlo()
        server.login(sender_email, sender_passwrd) 
        server.sendmail(sender_email, receiver_emails, message)

    except Exception as e:
        error_msg = "Problem sending email." + e
        utils.append_to_log(date, error_msg, LOGS_PATH + "nextdoor.err")
    else:
        success_msg = "Task completed successfully! Check your email."
        utils.append_to_log(date, success_msg, LOGS_PATH + "nextdoor.log")
        print(success_msg + "\n")
    finally:
        server.quit()
