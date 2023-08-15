import pandas as pd
import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def summarize_raw_data(data: dict) -> None:
    """
    Print summary statistics from dictionary of raw post data.
    """

    print("Number of posts: {}".format(len(data)))
    print("Number of important posts: {}".format(len({key: val for key, val in data.items() if val['Interacted']})))

    towns = [val['Location'] for val in data.values()]
    authors = [val['Author'] for val in data.values()]

    print("Number of unique hometowns: {}".format(len(set(towns))))
    print("Number of unique authors: {}".format(len(set(authors))))


def preprocess_historical_dataset(data: dict) -> pd.DataFrame:
    """
    Preprocess dictionary of raw post training data and return a filtered dataframe.

    DataFrame Columns
    -- id: ID of the post 
    -- Text: Filtered text of the post
    -- Age: Number of days since post's origination (float)
    -- NumReactions: Current number of users who reacted to the post (int)
    -- NumComments: Current number of comments left on the post (int)
    -- Interacted: Whether or not the subject has reacted/commented on the post (boolean)
    """

    df = pd.DataFrame.from_dict({i: data[i] for i in data.keys()}, orient='index')

    df['Text'] = df['Text'].apply(preprocess_text)
    df['Age'] = df['Age'].apply(convert_to_days)

    print(df)

    return df


def preprocess_text(text: str) -> str:
    """
    Perform text preprocessing on the contents of all posts. 
    """
    
    digit_pattern = re.compile(r'\d+')
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

    # Convert text to lowercase
    text = text.lower()

    # Remove punctuations from text
    text = ' '.join(text.translate(str.maketrans('', '', string.punctuation)).split())

    # Remove whitespace from text
    text = text.strip()

    # Remove all patterns from text
    for pattern in [digit_pattern, url_pattern, emoji_pattern]:
        text = pattern.sub(r'', text)

    # Remove stopwords (i.e. the, an, a, etc.) from text
    stop_words = set(stopwords.words('english'))
    stop_words = " ".join([word for word in str(text).split() if word not in stop_words])

    # Apply stemming to reduce words to their stem form
    stemmer = PorterStemmer()
    text = " ".join([stemmer.stem(word) for word in text.split()])

    # Remove whitespace from text
    text = text.strip()

    return text


def convert_to_days(age: str) -> float:
    """
    Convert post age from Nextdoor's format to a decimal number of days.

    Examples: 7 days ago => 7
              18 hours ago => 0.75 
    """

    items = age.split(" ")

    # Remove any "edited" tags
    if len(items) == 4:
        items.pop(0)

    num, identifier = items[0], items[1]

    if re.search('day', identifier):
        return float(num)
    
    if re.search('hr', identifier):
        return float(num)/24
    
    if re.search('min', identifier):
        return float(num)/3600
    
    return None






    





