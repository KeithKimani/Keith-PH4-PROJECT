# Twitter Sentiment Analysis on Apple and Google Products



## Overview and Business Understanding

Apple and Google are tech giants competing across multiple markets, including smartphones, tablets, and smart home devices, with contrasting approaches. Apple emphasizes a closed, premium ecosystem, while Google focuses on accessibility and cross-platform solutions. 

To aid stakeholders like marketers, product teams, and retailers, a proposed solution involves conducting a Twitter sentiment analysis on Apple and Google products. This analysis will classify tweets as positive, negative, or neutral and identify consumer sentiment trends. The insights aim to enhance marketing strategies, refine product development, and improve customer engagement, ultimately strengthening brand loyalty and product offerings.

## Problem Statement

**Group Six Company** recognizes the value of understanding consumer opinions shared on social media. Tasked by Apple, a global tech leader, we are analyzing over 9,000 human-annotated tweets to classify sentiment as positive, negative, or neutral using **Natural Language Processing (NLP)**. The project aims to uncover actionable insights that will inform Apple’s marketing and product development strategies. By understanding customer perceptions, we aim to help Apple refine its communication, enhance satisfaction, and maintain its leadership in the tech industry.



## Objective

### Main Objective

 - Build a Natural Language Processing (NLP) model that can rate the sentiment of a Tweet based on its content. 



### Secondary Objectives

 - Analyze and Compare Sentiment for Apple vs. Google Products
 - Identify Key Drivers of Positive and Negative Sentiment
 - Monitor and Track Sentiment Trends Over Time
 
 - Gather insights into customer preferences, opinions, and emerging trends
 


## Metrics of Success

- **Accuracy**: Measures the proportion of correctly classified tweets. Useful for overall performance but less effective for imbalanced datasets.
   1) **Formula**: (True Positives + True Negatives)/Total Samples
   2) **Goal**: Achieve at least 85% accuracy on a balanced test dataset.
 

- **Recall**: Focuses on identifying all actual positive cases.
   1) **Formula**: True Positives/(True Positives + False Negatives)
   2) **Importance**: Critical when false negatives are costly.
 

- **Precision**: Evaluates the proportion of correctly predicted positive cases out of all positive predictions.
   1) **Formula**: True Positives / (True Positives + False Positives)
   2) **Importance**: Key when false positives are problematic.
 

- **F1 Score**:Balances precision and recall as their harmonic mean.
   1) **Formula**: 2 ×(Precision × Recall) /(Precision + Recall)
   2) **Goal**: Target an F1-score of 0.85 or higher for each sentiment category, ensuring balanced performance. 
   3) **Importance**: Effective for imbalanced datasets where both false positives and false negatives matter.
    
# Data Understanding
The Twitter Dataset used in in this analysis has been sourced from CrowdFlower accessible through the link below: https://data.world/crowdflower/brands-and-product-emotions)

It contains roughly 9,000 Tweets which have been evaluated for sentiment by human raters, with the majority focusing on products from Apple and Google. Each tweet has been classified as either positive, negative or neutral. The dataset was collected during the 2013 South by SouthWest (SXSW) conference which is known for showcasing the latest technology. As such, it not only provides a unique setting to effectively capture the real time consumer reactions to brands but also provides an environment where customers can compare products from leading companies with reduced individual biases.

# Import relevant libraries
We will begin by importing all the relevant libraries that we will be using for this analysis. These are provided below:
```python
#Importing all the relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder



import nltk
nltk.download('wordnet')
from nltk.tokenize import RegexpTokenizer
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk import TweetTokenizer
import re
import unicodedata
import string
from collections import Counter



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

from xgboost import XGBClassifier
```
### Load Data

We will proceed with loading the data from our data file  `judge-1377884607_tweet_product_company.csv` 
We will call it `df`

```python
# Importing data_understanding class from DataFunctions script file 
# DataFunctions contains all the functions to be used in the notebook

from DataFunctions import data_understanding as du

df = du("judge-1377884607_tweet_product_company.csv")
```
### Checking first and last five rows

Next step is to check the first and last five rows to get an idea of how our data looks like.  

```python
#Checking first five rows

du.first_rows(df)
```
```python
#Checking last five rows

du.last_rows(df)
```
From the output above, our data is made up of 3 columns listed as follows:

- `tweet_text` - Contains a tweet alongside the username
- `emotion_in_tweet_is_directed_at` - Shows the product being referred to by the tweet
- `is there_an_emotion_directed_at_a_brand_or_product` - Highlights the sentiment i.e. positive, negative etc 

In addition, we observe that some of the tweets are in a language other than English as seen by the tweet in the last row (`row 9092`). 

Also, `emotion_in_tweet_is_directed_at` column appears to have null/missing values. This needs to be addressed as part of data preparation before proceeding with the analysis.

### Checking shape
Based on the findings above, let's have a look at the dimensions of the data we are dealing with. 

The data is made up of 9093 rows and 3 columns as shown below:
```python
#Checking the shape of the data

du.data_shape(df)
```

### Checking info
Let's dive deeper into our data. From the output below we observe that all the data columns are `text` which is what we expect. In addition, `tweet_text` is missing one entry while `emotion_in_tweet_directed_at`has more than 5000 missing entries. This is consistent with our observations thus far.

```python
#Checking the info on the data, specifically datatypes and missing values

du.data_info(df)
```
### Description of data

Finally, let's have a breakdown of the summary statistics of the data. 

```python
#Checking summary statistics of the data

du.data_description(df)
```
From the output above,  we observe that `tweet_text`  has the highest number of unique values with the most frequent occuring being 5. On the other hand, `is_there_an_emotion_directed_at_a_brand_or_product` only has 4 unique values with the most frequent appearing entry,which corresponds to having no emotion toward brand or product, appearing more than 5000 times. Finally, in terms of products (`emotion_in_tweet_is_directed_at`), iPad was the most frequent appearing category in 9 unique values.

## Data Preparation

### Data cleaning

Next step is to clean our data in preparation for analysis. Steps taken include:

- Renaming columns
- Handling Duplicates
- Handling missing values
- Merging sentiments
- Text data preparation

#### Renaming the columns

As a first step in our data preparation, we will rename the columns so that they are much easier to work with going forward. The columns will be renamed as follows:

- `tweet_text` : **Tweet**
- `emotion_in_tweet_is_directed_at`: **Product_Name**
- `is_there_an_emotion_directed_at_a_brand_or_product` : **Sentiment Type**

```pyhton
#renaming columns 
df.data = df.data.rename(columns = {'tweet_text': 'Tweet', 
                         'emotion_in_tweet_is_directed_at': 'Product_Name', 
                         'is_there_an_emotion_directed_at_a_brand_or_product': 'Sentiment_Type'})
du.first_rows(df)
```
#### Check missing values and duplicates

Next step is to check for missing values and duplicates.

```python
#Checking missing values and duplicates

du.check_missing_data_and_duplicates(df)
```

From the output above,  we have 5802 missing values in the `Product_Name` column and 1 missing entry in the `Tweet` column. 

In addition, we have 39 duplicates in our dataset.


#### Handling duplicates

We will start off by removing the duplicates so that it doesn't skew our analysis.

```python
#dropping duplicates
df.data.drop_duplicates(inplace=True)
```

```python
#Check if the values have been updated
du.check_missing_data_and_duplicates(df)
```

#### Handling missing values

Next, we will remove the one missing tweet from our analysis as we believe it will have a low impact on our analysis

```python
#removing missing tweet in the datset

df.data.dropna(subset=['Tweet'], inplace=True)
```

```python
#Checking if the value has been updated
du.check_missing_data_and_duplicates(df)
```

We can now proceed with dealing with the remaining null values in the` Product_Name column`. We will create a new column `Product` to assign apple products to Apple and Google products and fill in the missing values with `Unknown`

```python
# Categorize either Apple or Google products and fill in missing values with Unknown
# Define the mapping
product_mapping = {
    'iPad': 'Apple',
    'Apple': 'Apple',
    'iPad or iPhone App': 'Apple',
    'iPhone': 'Apple',
    'Other Apple product or service': 'Apple',
    'Google': 'Google',
    'Other Google product or service': 'Google',
    'Android App': 'Google',
    'Android': 'Google'
}

# Apply the mapping to create a new 'Product' column
df.data['Product'] = df.data['Product_Name'].map(product_mapping)

# Fill missing or unmapped values with 'Unknown'
df.data['Product'].fillna('Unknown', inplace=True)
```

```python
#Checking that the changes have been updated
df.data["Product"].value_counts()
```

Next we define a function below that takes in values from the created `Product` column as well `tweet` column. If the product is unknown from the `Product` column , the function seeks to look into the tweet to derive it.This will then be stored in the column `Brand`  

```python
# Function that finds a brand based on both the Product and tweet columns
def find_brand(Product, Tweet):
    """
    This function is designed to take build a brand column based on the Product description. 
    After, the function will look at the Tweets and determine a brand for rows with no brand determined.
    
    Product -  a column the function is working on
    Tweet - a column the function is working on
    """
    brand = 'Unknown' #Labeling brand as Unknown
    if ((Product.lower().__contains__('google')) or (Product.lower().__contains__('android'))): #Labeling Google
        brand = 'Google' #Unless tweet contains google or android
    elif ((Product.lower().__contains__('apple')) or (Product.lower().__contains__('ip'))): #Labeling Apple
        brand = 'Apple' #Unless tweet contains apple or ip
    
    if (brand == 'Unknown'): 
        lower_tweet = Tweet.lower() #Making tweet lowercase
        is_google = (lower_tweet.__contains__('google')) or (lower_tweet.__contains__('android')) #Undetermined google
        is_apple = (lower_tweet.__contains__('apple')) or (lower_tweet.__contains__('ip')) #Undetermined apple
        
        if (is_google and is_apple): #if it has both identifiers in the tweet
            brand = 'Both' #Labeling brand as both
        elif (is_google):
            brand = 'Google' #Labeling brand as Google
        elif (is_apple):
            brand = 'Apple' #Labeling brand as Apple
    
    return brand

df.data['Brand'] =df.data.apply(lambda x: find_brand(x['Product'], x['Tweet']), axis = 1) #Applying function to column
df.data['Brand'].value_counts() #Reviewing value counts of each class within brand
```

#### Merge sentiments
The `Sentiment_Type` labels could be merged so that we have three major distinct classes. As there are few sentiments labelled "I can't tell", we will merge this with the "neutral" sentiment assuming both evoke the same sentiment.

```python
#checking values in the Sentiment column
df.data["Sentiment_Type"].value_counts()
```

```python
#Remapping these values into three major distinct classes
remap_sentiment = {
    "No emotion toward brand or product": "neutral",
    "Positive emotion": "positive",
    "Negative emotion": "negative",
    "I can't tell": "neutral"
}

df.data["Sentiment_Type"] = df.data["Sentiment_Type"].apply(lambda x: remap_sentiment[x])
```

```python
#checking that mapping has been updated
df.data["Sentiment_Type"].value_counts()
```

### Text Data Preparation and Analysis

Now that we have all the data needed in the `Brand` and `Sentiment_Type` columns, we will now proceed with exploring aand analyzing the text data in the `Tweet` column using Natural Language Processing techniques.

To begin with we will change all the tweets to lower case.

```python
# shift all tweets to lower case
df.data["Tweet"] = df.data["Tweet"].str.lower()
```

Next we make a list of the tweets.

```python
# make list of all tweet texts
tweets = df.data["Tweet"].to_list()

#Displays the first five tweets
tweets[0:5]
```

Next step is to create a list of all the tokens. To help us achieve this, we'll use a tokenizer `TweetTokenizer` that is specifically designed to dissect tweets from Twitter.

```python
# Create a tokenizer
tokenizer = TweetTokenizer(
    preserve_case=False,
    strip_handles=True
)

# create list of tokens from data set
tokens = tokenizer.tokenize(','.join(tweets))

tokens[:20]
```

The tokens created still have hashtags which need to be removed.

```python
# Function to remove hashtags and accents

def remove_accents(word):
    """
    Remove accents from a given word.
    """
    return ''.join(
        char for char in unicodedata.normalize('NFD', word)
        if unicodedata.category(char) != 'Mn'
    )

# Process tokens
tokens = [remove_accents(word) for word in tokens if not word.startswith('#')] + \
         [remove_accents(word[1:]) for word in tokens if word.startswith('#')]

tokens[:20]
```

We will now remove the punctuation marks before lemmatizing.

```python
# Remove punctuation marks
tokens_cleaned = [
    word.translate(str.maketrans('', '', string.punctuation)) for word in tokens if word.translate(str.maketrans('', '', string.punctuation))
]

tokens_cleaned[:20]
```

We will proceed with lemmatizing the text. We will use the implementation of lemmatization that incorporates part-of-speech (POS) tagging for better accuracy. By providing the correct POS tags, the lemmatizer can produce more accurate base forms.


```python
# Function to map NLTK POS tags to WordNet POS tags

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun
```

```python
# Initiate lemmatizer
lemmatizer = WordNetLemmatizer()

# Perform POS tagging and lemmatize based on POS
tokens_lemmatized = [
    lemmatizer.lemmatize(word, get_wordnet_pos(pos))
    for word, pos in pos_tag(tokens_cleaned)
]
```

We can now generate the frequency distributions of `tokens_lemmatized`

```python
freq_dist = FreqDist(tokens_lemmatized)
freq_dist
```

```python
# Visualizing the top 20 word frequency associated with the tweets  
from matplotlib.ticker import MaxNLocator

def visualize_top_20(freq_dist, title):

    # Extract data for plotting
    top_20 = list(zip(*freq_dist.most_common(20)))
    tokens = top_20[0]
    counts = top_20[1]

    # Set up plot and plot data
    fig, ax = plt.subplots(figsize =(12,8))
    ax.bar(tokens, counts)

    # Customize plot appearance
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="x", rotation=90)
    
visualize_top_20(freq_dist, "Top 20 Word Frequency for Tweets")
```

From the bargraph above, we observe that the top occuring words are dominated by stop words

```python
# Load stopwords for English
stop_words = set(stopwords.words('english'))


# Remove stopwords from the tokens
tokens_finalised = [word for word in tokens_lemmatized if word.lower() not in stop_words]
```

```python
freq_dist_2 = FreqDist(tokens_finalised)
freq_dist_2
```

```python
# Visualizing the top 20 word frequency associated with the tweets  
from matplotlib.ticker import MaxNLocator


def visualize_top_20(freq_dist_2, title):

    # Extract data for plotting
    top_20 = list(zip(*freq_dist_2.most_common(20)))
    tokens = top_20[0]
    counts = top_20[1]

    # Set up plot and plot data
    fig, ax = plt.subplots(figsize =(12,8))
    ax.bar(tokens, counts)

    # Customize plot appearance
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="x", rotation=90)
    
visualize_top_20(freq_dist_2, "Top 20 Word Frequency for Tweets")
```

We still have words such as sxsw, which is the name of the festival as well as twitter words such as link and RT that need to be removed. In addition, there is a number 2 which will be removed given that it's most likely referring to the Ipad 2 which was been launched at the time as we will treat all ipads as the same. Finally, there is a special character "\x89" that appears in our data as well as single value entries. we will use `Regular Expressions`  to remove the special characters, numbers and all single entries from our list.

```python
#Introduing RegEx pattern to remove \,numbers, single entries
pattern = r"^\\|^.$|^\d+$"

# Remove stopwords from the tokens
tokens_finalised= [word for word in tokens_finalised if not re.match(pattern, word)]

# Remove sxsw, RT and link

stop_list = ["sxsw", "link", "rt"]

tokens_final = [word for word in tokens_finalised if word not in stop_list]
```

```python
freq_dist_3 = FreqDist(tokens_final)
freq_dist_3
```

```python
# Visualizing the top 20 word frequency associated with the tweets  
from matplotlib.ticker import MaxNLocator


def visualize_top_20(freq_dist_3, title):

    # Extract data for plotting
    top_20 = list(zip(*freq_dist_3.most_common(20)))
    tokens = top_20[0]
    counts = top_20[1]

    # Set up plot and plot data
    fig, ax = plt.subplots(figsize =(12,8))
    ax.bar(tokens, counts)

    # Customize plot appearance
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="x", rotation=90)
    
visualize_top_20(freq_dist_3, "Top 20 Word Frequency for Tweets")
```

Much better now. Our text data now contains ipad as the most fequent word followed by google. We can now create a function below that does all these steps to the `Tweet` column of our dataframe listed as follows:

 - instantiate tokenizer
 - remove hashtags and accents
 - remove punctuation marks
 - map NLTK POS tags to WordNet POS tags
 - instantiate lemmatizer
 - Perform POS tagging and lemmatize based on POS
 - Remove stopwords from the tokens, words from our stop list and special characters

```python
def custom_tokenize_2(text):
    # Ensure input is a string
    if isinstance(text, list):
        text = ' '.join(map(str, text))  # Convert list to a single string
    elif not isinstance(text, str):
        raise ValueError(f"Expected string or list, but got {type(text)}")

    # Instantiate tokenizer
    tokenizer = TweetTokenizer(
        preserve_case=False,
        strip_handles=True
    )
    
    # Create list of tokens from the text
    tokens = tokenizer.tokenize(text)
    
    # Process tokens
    tokens = [remove_accents(word) for word in tokens if not word.startswith('#')] + \
             [remove_accents(word[1:]) for word in tokens if word.startswith('#')]
    
    # Remove punctuation marks
    tokens_cleaned = [word.translate(str.maketrans('', '', string.punctuation)) 
                      for word in tokens if word.translate(str.maketrans('', '', string.punctuation))]
    
    # Remove stop words
    tokens_finalised = [word for word in tokens_cleaned if word.lower() not in stop_words]
    
    # Introduce RegEx pattern to remove \, numbers, single entries
    pattern = r"^\\|^.$|^\d+$"
    tokens_finalised = [word for word in tokens_finalised if not re.match(pattern, word)]
    
    # Remove specific words like "sxsw", "RT", and "link"
    stop_list = ["sxsw", "link", "rt"]
    tokens_final = [word for word in tokens_finalised if word not in stop_list]
    
    # Map NLTK POS tags to WordNet POS tags
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # Default to NOUN
    
    # Instantiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Perform POS tagging and lemmatize based on POS
    tokens_lemmatized = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tag(tokens_final)
    ]
    
    return tokens_lemmatized
```

```python
df.data['Tweet_tokens'] = df.data['Tweet'].apply(custom_tokenize_2)
```

Great! Our dataset is now ready for Exploratory Data Analysis.

```python
#Preview of our finalised dataset

df.data.style.set_properties(**{'text-align': 'left'})
```

## Exploratory Data Analysis

### Sentiment Distribution

```python
# Count sentiment occurrences
sentiment_counts = final_df['Sentiment_Type'].value_counts()

# Create a pie chart or bar graph for sentiment comparison
plt.figure(figsize=(12, 8))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment Type')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()
```

**Observation**: From the sentiment distribution above, we can observe that neutral has the highest count followed by positive sentiments with negative having the lowest counts. This implies  that fewer users expressed dissatisfaction or unfavourable opinions while majority of the users neither expressed strong positivity or negativity.

### Distribution of brands

```python
# Count sentiment occurrences
brand_counts = final_df['Brand'].value_counts()

# Create a pie chart or bar graph for sentiment comparison
plt.figure(figsize=(12, 8))
brand_counts.plot(kind='bar', color=['green', 'red', 'gray'])
plt.title('Distribution of brands')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()
```
**Observation**: From the distribution above, Apple had the highest response counts with more than 5,000 tweets directed at it. Google, came in second with roughly 3,000 tweets mentioning it. Interestingly, tweets which didn't mention any of the brands were higher than those which mentioned both. Apple had a much better response compared to Google. We can assume that they had more products on display or their products were slightly better hence eliciting more better response.

### Sentiment Distribution by Brand

```python
# Group by Brand and Sentiment_Type to count occurrences
sentiment_counts = final_df.groupby(['Brand', 'Sentiment_Type']).size().unstack(fill_value=0)

# Plotting the bar graph
sentiment_counts.plot(kind='bar', stacked=False, figsize=(12, 8))

# Adding titles and labels
plt.title('Sentiment Distribution by Brand')
plt.xlabel('Brand')
plt.ylabel('Count of Sentiments')
plt.xticks(rotation=0)
plt.legend(title='Sentiment Type')

# Show the plot
plt.show()
```
**Observation**: From the graph above, it fairly consistent among all the brands that neutral sentiment supersedes all the other sentiments. In the case of Apple and Google, positive sentiments are higher than negative sentiments. Also, the proportions of neutral sentiment to positive or negative sentiments is much higher in the `Unknown` category. Apple faired much better in terms of positive sentiments when compared to Google. 

### Top Words per Sentiment

```python
# Flatten the list of words grouped by Sentiment_Type
plt.rcParams['font.family'] = 'Arial'

def get_word_frequencies(final_df, sentiment):
    tokens = final_df[final_df['Sentiment_Type'] == sentiment]['Tweet_tokens']
    all_words = [word for tokens_list in tokens for word in tokens_list]
    return Counter(all_words)

# Define sentiments
sentiments = final_df['Sentiment_Type'].unique()

# Plot the top words for each sentiment type
for sentiment in sentiments:
    # Get word frequencies for this sentiment
    word_freq = get_word_frequencies(final_df, sentiment)
    
    # Get the top 10 most common words
    top_words = word_freq.most_common(10)
    
    # Create a bar chart
    words, counts = zip(*top_words)
    plt.figure(figsize=(10, 5))
    plt.bar(words, counts, color='skyblue')
    plt.title(f'Top Words in {sentiment} Sentiment')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()
```
**Observation**: From the graphs above, ipad is the most common word amongst the positive and negative sentiments. On the other hand, google is the most common word amongst people who had a neutral sentiment. This pattern may indicate that Apple's products, particularly the iPad, evoked stronger emotional reactions (positive or negative), whereas Google was discussed in a more neutral tone.

## Feature Engineering


### Convert tokenized data to strings

As part of preprocessing, we will start by converting the lemmatized tokens into strings storing them in a new column `Processed_Tweets

```python
# Convert the tokenized column to a single string per row
final_df['Processed_Tweets'] = final_df['Tweet_tokens'].apply(lambda tokens: ' '.join(tokens))
final_df.head()
```

### Create new columns and dataframe

We will create a new dataframe `binary_df` that captures only the `negative` and `positive` sentiments. We would like to perform this analysis in two steps:
1. **Binary classification** - Analyses only Positive and Negative sentiments
2. **Multiclass classification** - Analyses Positive, Negative and Neutral sentiments

In addition, a new_column `y_binary` has been created in this data with `positive` sentiments being labelled 1 and `negative` sentiments labelled 0.

```python
#Perform feature engineering
# Filter the dataset for only positive and negative sentiments
binary_df = final_df[final_df['Sentiment_Type'].isin(['positive', 'negative'])].copy()

# Prepare target variable (1 for positive, 0 for negative)
binary_df["y_binary"] = binary_df['Sentiment_Type'].apply(lambda x: 1 if x == 'positive' else 0)

binary_df
```

### Check for imbalance

Given, the binary classification problem we will start with, let's check for class imbalance in the target variable.
As can be observed below, we do have a clear imbalance in our target variable`y_binary`. Therefore if we had a model that always picked people that had a positive sentiment (majority class) then we would expect an accuracy score of around 83%. This clas imbalance issue will be looked at as part of building the model.

```python
#Checking for imbalance in the target variable

print(binary_df["y_binary"].value_counts())
print("\n-----------------")
print(binary_df["y_binary"].value_counts(normalize=True))
```

### Perform a train-test split

Create variables `X_train`, `X_test`, `y_train`, and `y_test` using train_test_split with `X`, `y`, `test_size` =0.3, and `random_state`=42.

```python
#Define our X and y variables

X= binary_df["Processed_Tweets"]
y =binary_df["y_binary"]

#Perform train_test_split

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.3, random_state =42)
```

## Modelling 
### Binary Classification

We will begin with a binary classification with the focus being on only `positive` and `negative` sentiments

### Baseline Model
Our baseline model will be the Multinomial Naive Bayes (NB) classifier. Given the size of our dataset, this model is particularly well-suited due to its efficiency and effectiveness in handling text data represented as word frequencies or TF-IDF features. It assumes a multinomial distribution for features, which aligns well with the nature of text, where word occurrences are critical indicators of sentiment. Its computational simplicity makes it ideal for medium-sized datasets  allowing for quick training and evaluation. Furthermore, it performs well with sparse data, typical in text analysis, where most words in the vocabulary do not appear in every document. The probabilistic nature of Multinomial NB also provides interpretable outputs, enabling insights into which words are most associated with each sentiment class. Despite its simplicity, it can achieve competitive results and serves as a strong baseline for sentiment analysis.

### Create a pipeline

Create a pipeline that TF-IDF vectorizes text input and then feeds it into a Multinomial Naive Bayes classifier with `max_features` =10 as a start. This pipeline has been saved as a variable **nlp_pipe**.

```python
# Create a pipeline iwth TF-IDF vectorizer and MultiNomial Naive Bayes Classifier

nlp_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10)),
    ("classifier", MultinomialNB())
  
])
```

```python
#Fit the pipe to the training data and predict using the test data
nlp_pipe.fit(X_train,y_train)
y_pred_baseline = nlp_pipe.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred_baseline)
precision = precision_score(y_test, y_pred_baseline, average='weighted', zero_division=0)
recall = recall_score(y_test,y_pred_baseline, average='weighted')
f1 = f1_score(y_test,y_pred_baseline, average='weighted')

# Print results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
```
Our baseline model performs slightly better with an accuracy of 85% when compared to selecting the majority class most of the time. However, with the low precision score, it seems our modelstill misclassifies most of the negative sentiments.

### Performing  a Grid Search for best parameters
Let's perform a grid search to modify our parameters in the TF-IDF vectorizer as well as the classifier and check if they will give us much better results.

# Define the parameter grid for GridSearchCV
param_grid = {
    'tfidf__max_features': [10, 50, 100, None], # No of features to include when building the TF-IDF representation
    'tfidf__ngram_range': [(1, 1), (1, 2)], #Range of n-grams (sequence of words) considered in the TF-IDF Vectorization process
    'classifier__alpha': [0.1, 1, 10] #Smoothing parameter for the Naive Bayes Classifier
}

# Instantiate GridSearchCV
grid_nb = GridSearchCV(nlp_pipe, param_grid=param_grid, scoring="accuracy", cv=5)

# Fit the grid search
grid_nb.fit(X_train, y_train)

# Get the best estimator
best_model = grid_nb.best_estimator_

# Display the best parameters
print("Best Parameters:", grid_nb.best_params_)
print("Best Score:", grid_nb.best_score_)

### Tuned Mutinominal NB model  with best parameters as per the GridSearch
Our tuned NB model will be the `best_model` with the best parameters as per the grid search. We will now refit the train data to this model and see if our evaluation scores improve.

```python
# Refit to train
best_model.fit(X_train,y_train)

# Test set predictions and scores
y_best_pred = best_model.predict(X_test)
tuned_test_acc = accuracy_score(y_test,y_best_pred)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_best_pred)
precision = precision_score(y_test, y_best_pred, average='weighted', zero_division=0)
recall = recall_score(y_test,y_best_pred, average='weighted')
f1 = f1_score(y_test,y_best_pred, average='weighted')

# Print results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
```
All our evaluation scores have improved when compared to the baseline model scores. Precision and Recall both have gone up to roughly 88% leading to an improvement in the F1-score to 86%. This model does a much better job at classifying.

### Tuned model  - Reducing the class imbalance 

Remember as part of pre-processing we had a look at the target variable `y_binary` and it was imbalanced. We will try and use `SMOTE` oversampling technique to handle the imbalance and check if our model scores improve. To do so we will add this into our pipe and rename this variable **nlp_pipe_2**

```python
# Create SMOTE object
smote = SMOTE(random_state=42)

#Pipe
nlp_pipe_2  = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10)),
    ("smote",SMOTE(random_state=42)),
    ("classifier", MultinomialNB())
  
])

nlp_pipe_2.fit(X_train,y_train)
y_pred_smote = nlp_pipe_2.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred_smote)
precision = precision_score(y_test, y_pred_smote, average='weighted', zero_division=0)
recall = recall_score(y_test,y_pred_smote, average='weighted')
f1 = f1_score(y_test,y_pred_smote, average='weighted')

# Print results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
```
Our evaluation scores have gone down when we fit the baseline model parameters to the oversmpled data. Only precision is within our range at 80% while the recall is at 53%. This means that the model will misclassify a lot of the positive sentiments as negative which is ideally what we do not want.

### Tuned MultiNomial NB model with reduced class imbalance incorporating best parameters
Given the outcome above, we will perform a similar grid search to find the optimal parameters

```python
# Define the parameter grid for GridSearchCV
param_grid = {
    'tfidf__max_features': [10, 50, 100, None], # No of features to include when building the TF-IDF representation
    'tfidf__ngram_range': [(1, 1), (1, 2)], #Range of n-grams (sequence of words) considered in the TF-IDF Vectorization process
    'classifier__alpha': [0.1, 1, 10]       #Smoothing parameter for the Naive Bayes Classifier
}

# Instantiate GridSearchCV
grid_nb_smote = GridSearchCV(nlp_pipe_2, param_grid=param_grid, scoring="accuracy", cv=5)

# Fit the grid search
grid_nb_smote.fit(X_train, y_train)

# Get the best estimator
best_model_smote = grid_nb_smote.best_estimator_

# Display the best parameters
print("Best Parameters:", grid_nb_smote.best_params_)
print("Best Score:", grid_nb_smote.best_score_)
```

```python
#Refit to train
best_model_smote.fit(X_train,y_train)

# Test set predictions and scores
y_best_pred_smote = best_model_smote.predict(X_test)

# Calculate evaluation metrics
accuracy_smote = accuracy_score(y_test, y_best_pred_smote)
precision_smote = precision_score(y_test, y_best_pred_smote, average='weighted', zero_division=0)
recall_smote = recall_score(y_test,y_best_pred_smote, average='weighted')
f1_smote = f1_score(y_test,y_best_pred_smote, average='weighted')

# Print results
print("Accuracy:", accuracy_smote)
print("Precision:", precision_smote)
print("Recall:", recall_smote)
print("F1-score:", f1_smote)
```

This performs much better but the `best_model` had the better evaluation results. `Precision` and `Recall` are slightly lower leading to a reduced `F1-score` of 0.85.  Thus, even though we performed oversampling, it did not improve the results of our model as expected. For the Multionomial NB classifier, `best_model` is our best model so far. A confusion matrix on the model results is shown below:

```python
#Confusion matrix on best model using the test data

cnf_matrix = confusion_matrix(y_test,y_best_pred)

# Create a figure with specified size
fig, ax = plt.subplots(figsize=(10, 8))  # Adjust size as needed

# Create and plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix)
disp.plot(ax=ax);
```
From the confusion matrix above, the model performs well at correctly classifying positive instances(890). There is a moderate number of samples misclassified as positive (106), suggesting room for improvement in precision. The model rarely misses positive samples (13), which is good for recall. Fewer instances (53) are correctly classified as negative, indicating the negative class might be under-represented or harder to classify.

### Neural network

Based on the results we have observed on the MultiNomial NB model, we will fit a neural network to check if it does a better job at the binary classification.

```python
# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

# Define the neural network model
model = Sequential([
    Dense(128, input_dim=X_train_tfidf.shape[1], activation='relu'),  # Input layer
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(64, activation='relu'),  # Hidden layer
    Dropout(0.3),  # Dropout
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_tfidf, y_train, epochs=10, batch_size=16, validation_split=0.2, verbose=1)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_tfidf, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Predict on the test set
y_pred = model.predict(X_test_tfidf)
y_pred_binary = (y_pred > 0.5).astype(int).flatten()

# Calculate evaluation metrics
accuracy_nn = accuracy_score(y_test, y_pred_binary)
precision_nn = precision_score(y_test, y_pred_binary, average='weighted', zero_division=0)
recall_nn = recall_score(y_test,y_pred_binary, average='weighted')
f1_nn = f1_score(y_test,y_pred_binary, average='weighted')

# Print results
print("Accuracy:", accuracy_nn)
print("Precision:", precision_nn)
print("Recall:", recall_nn)
print("F1-score:", f1_nn)
```
### Tuned neural network


Tuning the neural network to improve its perfomance. As such we will be changing the parameters as follows:
    
- Change the `max_features` of the `TfIdf vectorizer` to 5000 to capture more information about the dataset
- Increase `hidden Layers`: Add one `hidden layer` to capture complex patterns in the data.
- Increase `epoch size` to 50 to allow the model to learn better while ensuring it does not overfit.
- Changed `batch size` to 8 to allow the model to run on a much smaller batch.

```python
# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

# Define the neural network model
model = Sequential([
    Dense(128, input_dim=X_train_tfidf.shape[1], activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),  # Additional hidden layer
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_tfidf, y_train, epochs=20, batch_size=8, validation_split=0.2, verbose=1)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test_tfidf, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Predict on the test set
y_pred = model.predict(X_test_tfidf)
y_pred_binary = (y_pred > 0.5).astype(int).flatten()

# Calculate evaluation metrics
accuracy_nn = accuracy_score(y_test, y_pred_binary)
precision_nn = precision_score(y_test, y_pred_binary, average='weighted', zero_division=0)
recall_nn = recall_score(y_test,y_pred_binary, average='weighted')
f1_nn = f1_score(y_test,y_pred_binary, average='weighted')

# Print results
print("Accuracy:", accuracy_nn)
print("Precision:", precision_nn)
print("Recall:", recall_nn)
print("F1-score:", f1_nn)
```

The results are almost similar to the neural network that has not been tuned. It's worth noting that the accuracy, precision, recall and F1 score observed here is slightly lower than what was observed by our best Multinomial NB model. This is expected as the size of our dataset is quite small. With more data, we believe the neural network will exhibit better reults.
Below is a confusion matrix of the neural network model results:

```python
cnf_matrix_nn = confusion_matrix(y_test, y_pred_binary)


# Create a figure with specified size
fig, ax = plt.subplots(figsize=(10, 8))  # Adjust size as needed

# Create and plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix_nn)
disp.plot(ax=ax);
```
Comparing the confusion matrix above to the MultiNomial NB classifie confusion matrix we note that the number of correctly classified negatives has increased, indicating better performance for the negative class.There are also fewer false positives in the current matrix, indicating better precision. However, the number of false negatives has increased, meaning the model is now missing more actual positive samples, reducing recall. Finally, the number of true positives has decreased, indicating a drop in correctly classified positives.

Therefroe the Neural Network model has become better at classifying negative samples, as seen in the increase in true negatives and decrease in false positives. On the other hand, the model's ability to correctly classify positive samples has decreased, evident from the higher false negatives and lower true positives. Thus, recall for the positive class has likely decreased while precision for the negative class has likely improved.

### Multiclass Classification:

Next we perform a similar analysis on a multiclass classification taking into account Positive, Negative and Neutral sentiments


### Preprocessing

#### Label Encode the Target Variable

Given we have three classes to analyze, we will label encode and store them in a new column `Sentiment_Label`. 

```python
#creaeting a multiclass df to be used in our analysis
multi_class_df = final_df.copy()
multi_class_df

# Instantiate LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the Sentiment_Type column
multi_class_df['Sentiment_Label'] = label_encoder.fit_transform(multi_class_df['Sentiment_Type'])

# Display the encoded values
print("Classes:", label_encoder.classes_)  # To see the mapping
multi_class_df['Sentiment_Label'].value_counts()
```

#### Defining our variables

```python
#Defining our X and y variables

X_multi= multi_class_df["Processed_Tweets"]
y_multi =multi_class_df["Sentiment_Label"]

#Perform train_test_split

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split (X_multi,y_multi, test_size=0.3, random_state =42)
```

### Baseline Model

Similarly,we will use a Multinomial NB classifier as our baseline model

```python
# Create a pipeline iwth TF-IDF vectorizer and MultiNomial Naive Bayes Classifier

nlp_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10)),
    ("classifier", MultinomialNB())
  
])

#Fit the pipe to the training data and predict using the test data
nlp_pipe.fit(X_train_multi,y_train_multi)
y_pred_multi_baseline = nlp_pipe.predict(X_test_multi)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test_multi, y_pred_multi_baseline)
precision = precision_score(y_test_multi, y_pred_multi_baseline, average='weighted', zero_division=0)
recall = recall_score(y_test_multi,y_pred_multi_baseline, average='weighted')
f1 = f1_score(y_test_multi,y_pred_multi_baseline, average='weighted')

# Print results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
```
`Accuracy` and `Recall` are roughly 64% with the `Precision` slightly lower at 57%. As a result the f1 score is a bit low at 0.51. This is way below our expected value of 85% across all the metrics so we will perform a grid search to tune the hyper parameters and see if we get better results

### Performing a Grid Search

```python
# Define the parameter grid for GridSearchCV
param_grid = {
    'tfidf__max_features': [10, 50, 100, None], # No of features to include when building the TF-IDF representation
    'tfidf__ngram_range': [(1, 1), (1, 2)], #Range of n-grams (sequence of words) considered in the TF-IDF Vectorization process
    'classifier__alpha': [0.1, 1, 10]       #Smoothing parameter for the Naive Bayes Classifier
}

# Instantiate GridSearchCV
grid_nb_multi = GridSearchCV(nlp_pipe, param_grid=param_grid, scoring="accuracy", cv=5)

# Fit the grid search
grid_nb_multi.fit(X_train_multi, y_train_multi)

# Get the best estimator
best_model_multi = grid_nb_multi.best_estimator_

# Display the best parameters
print("Best Parameters:", grid_nb_multi.best_params_)
print("Best Score:", grid_nb_multi.best_score_)
```

### Tuned MultiNomial Model using the best parameters from Grid search

```python
#Refit to train
best_model_multi.fit(X_train_multi,y_train_multi)

# Test set predictions and scores
y_best_pred_multi = best_model_multi.predict(X_test_multi)

# Calculate evaluation metrics
accuracy_multi = accuracy_score(y_test_multi, y_best_pred_multi)
precision_multi = precision_score(y_test_multi, y_best_pred_multi, average='weighted', zero_division=0)
recall_multi = recall_score(y_test_multi,y_best_pred_multi, average='weighted')
f1_multi = f1_score(y_test_multi,y_best_pred_multi, average='weighted')

# Print results
print("Accuracy:", accuracy_multi)
print("Precision:", precision_multi)
print("Recall:", recall_multi)
print("F1-score:", f1_multi)
```
Much better! The model performs much better. However, let's try a different model and see if we can even get better results.

### XGBoost

For this multi-class problem, we will try the `XGBoost` model as it's slightly recommended for dealing with datasets that are slightly larger than ours. Given the neural network did not improve our evaluation scores in our binary classification analysis, it is worth trying it to check if we get different results 


```python
# Create a pipeline with TF-IDF vectorizer and MultiNomial Naive Bayes Classifier

nlp_pipe_xgb = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10)),
    ("classifier",  XGBClassifier(random_state=42))
  
])

#Fit the pipe to the training data and predict using the test data
nlp_pipe_xgb.fit(X_train_multi,y_train_multi)
y_pred_multi_xgb = nlp_pipe_xgb.predict(X_test_multi)

# Calculate evaluation metrics
accuracy_xgb = accuracy_score(y_test_multi, y_pred_multi_xgb)
precision_xgb = precision_score(y_test_multi, y_pred_multi_xgb, average='weighted', zero_division=0)
recall_xgb = recall_score(y_test_multi,y_pred_multi_xgb, average='weighted')
f1_xgb = f1_score(y_test_multi,y_pred_multi_xgb, average='weighted')

# Print results
print("Accuracy:", accuracy_xgb)
print("Precision:", precision_xgb)
print("Recall:", recall_xgb)
print("F1-score:", f1_xgb)
```

```python
# Define the parameter grid
param_grid = {
    "tfidf__max_features": [10, 50, 100, None], # No of features to include when building the TF-IDF representation
    "tfidf__ngram_range": [(1, 1), (1, 2)], #Range of n-grams (sequence of words) considered in the TF-IDF Vectorization process
    "classifier__n_estimators": [50, 100, 200],  # Number of trees
    "classifier__max_depth": [3, 5, 7],  # Maximum depth of a tree
    "classifier__learning_rate": [0.01, 0.1, 0.2],  # Learning rate
}

# Define the pipeline
nlp_pipe_xgb = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10)),
    ("classifier", XGBClassifier(random_state=42))
])

# Create the GridSearchCV object
grid_search = GridSearchCV(
    nlp_pipe_xgb,
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring="accuracy",  # Use accuracy as the scoring metric
    verbose=1,  # Print progress
    n_jobs=-1  # Use all available cores
)

# Fit the GridSearchCV to the training data
grid_search.fit(X_train_multi, y_train_multi)

# Get the best pipeline
best_pipe_xgb = grid_search.best_estimator_

# Display the best parameters
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
```

```python
# Predict using the test data
y_pred_multi_xgb = best_pipe_xgb.predict(X_test_multi)

# Calculate evaluation metrics
accuracy_xgb = accuracy_score(y_test_multi, y_pred_multi_xgb)
precision_xgb = precision_score(y_test_multi, y_pred_multi_xgb, average='weighted', zero_division=0)
recall_xgb = recall_score(y_test_multi, y_pred_multi_xgb, average='weighted')
f1_xgb = f1_score(y_test_multi, y_pred_multi_xgb, average='weighted')

# Print results
print("Accuracy:", accuracy_xgb)
print("Precision:", precision_xgb)
print("Recall:", recall_xgb)
print("F1-score:", f1_xgb)
```
The XG boost model has perfomed much better than the Multinomial NB model in this regard. There is still scope for improvement given the accuracy is roughly 70% with the Precision and Recall being 67% and 70% respectively with more data and fine tuning 

```python
#Generate confusion matrix
cnf_matrix_xgb = confusion_matrix(y_test_multi,y_pred_multi_xgb)


# Create a figure with specified size
fig, ax = plt.subplots(figsize=(10, 8))  # Adjust size as needed

# Create and plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix_xgb)
disp.plot(ax=ax);
```
As per the confusion matrix above, the model correctly predicted 35 instances of Class 0 (Negative), 1483 instances of Class 1 (Neutral) and  373 instances of Class 2 (Positive). Class 1 (Neutral) has significantly more samples compared to Classes 0 (Negative) and 2 (Positive), leading to dominant predictions for Nuetral class. Misclassification is heavily skewed against the minority classes (positive and negative).Hence a high recall for Neutral class. The model performs well in identifying Neutral class (1483), likely due to its larger representation in the dataset but has challenges with minority Classes. ositive and Negative classes have higher rates of misclassification, with many instances being confused with the Neutral class. This suggests the model struggles with minority classes and may benefit from strategies to handle class imbalance.


## XG Boost with OverSampling

Based on the findings above, we will address the imbalance and check if we get a much better result 

```python
# Define the parameter grid
param_grid = {
    "tfidf__max_features": [10, 50, 100, None], # No of features to include when building the TF-IDF representation
    "tfidf__ngram_range": [(1, 1), (1, 2)], #Range of n-grams (sequence of words) considered in the TF-IDF Vectorization process
    "classifier__n_estimators": [50, 100, 200],  # Number of trees
    "classifier__max_depth": [3, 5, 7],  # Maximum depth of a tree
    "classifier__learning_rate": [0.01, 0.1, 0.2],  # Learning rate
}

# Define the pipeline
nlp_pipe_xgb_2 = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10)),
    ("smote",SMOTE(random_state=42)),
    ("classifier", XGBClassifier(random_state=42))
])

# Create the GridSearchCV object
grid_search_2 = GridSearchCV(
    nlp_pipe_xgb_2,
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring="accuracy",  # Use accuracy as the scoring metric
    verbose=1,  # Print progress
    n_jobs=-1  # Use all available cores
)

# Fit the GridSearchCV to the training data
grid_search_2.fit(X_train_multi, y_train_multi)

# Get the best pipeline
best_pipe_xgb_smote = grid_search_2.best_estimator_

# Display the best parameters
print("Best Parameters:", grid_search_2.best_params_)
print("Best Score:", grid_search_2.best_score_)
```

```python
# Predict using the test data
y_pred_multi_xgb_smote = best_pipe_xgb_smote.predict(X_test_multi)

# Calculate evaluation metrics
accuracy_xgb_smote = accuracy_score(y_test_multi, y_pred_multi_xgb_smote)
precision_xgb_smote = precision_score(y_test_multi, y_pred_multi_xgb_smote, average='weighted', zero_division=0)
recall_xgb_smote = recall_score(y_test_multi, y_pred_multi_xgb_smote, average='weighted')
f1_xgb_smote = f1_score(y_test_multi, y_pred_multi_xgb_smote, average='weighted')

# Print results
print("Accuracy:", accuracy_xgb_smote)
print("Precision:", precision_xgb_smote)
print("Recall:", recall_xgb_smote)
print("F1-score:", f1_xgb_smote)
```
We see an increase in the precision but a decline in both the accuracy and recall of the model once we oversample. As a result the F1 score has slightly reduced.

```python
#Generate confusion matrix
cnf_matrix_xgb_smote = confusion_matrix(y_test_multi,y_pred_multi_xgb_smote)


# Create a figure with specified size
fig, ax = plt.subplots(figsize=(10, 8))  # Adjust size as needed

# Create and plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix_xgb_smote)
disp.plot(ax=ax);
```
As a result of the SMOTE analysis, Class 0 (Negative) and Class 2 (Positive) both show slight improvements in true positive rates. Misclassification of  positive instances as neutral has decreased by 20 cases.
Class 1 (Neutral) sees a noticeable drop in correct classifications (1442 vs. 1483) and an increase in misclassifications into other classes. Misclassifications of Class 1 (Neutral) and Class 2 (Positive) increased significantly (+32 cases).
As a result, the current model slightly balances performance between Classes 0 (Negative) and 2 (Positive) while sacrificing performance for Class 1 (Neutral), indicating a shift in the focus of the model.


## Model Evaluation

#### Binary classification

- We attempted to oversample the data in order to deal with the class imbalance issue. This gave us worse results when we implemented it.

- Our efforts at tuning the models gave us better results when it came to the `MultiNomial NB classifier` but this wasn't the case for `Neural Networks`.

- `MultiNomial NB classifier` was much better when compared to the `Neural Network` when looking at the metrics. As a result `MultiNomial NB classifier` with no oversampling was chosen as the best model.

#### Multi- class classification

- Similarly, attempts to oversample the data in order to deal with the class imbalance issue, gave us worse results when we implemented it.

- The evaluation metrics were quite low in the multi class classification models when compared to the binary classification models

- Our efforts at tuning the models gave us better results for all the models used in this instance.

- `XG Boost` came out as the better model as opposed to the `MultiNomial NB classifier` in modelling the multiclass sentiments. `XG Boost` model with no oversampling  was chosen as the best model for this multi class sentiment analysis.

## Recommendation


## Conclusion


