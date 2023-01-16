# %%
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.dates as dates

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize

import seaborn as sns
from scipy.stats import shapiro

from wordcloud import WordCloud, STOPWORDS
# nltk.download('stopwords')
from nltk.corpus import stopwords

# %%
def preprocess(df):
    # Simple preprocessing - remove duplicates (hashtag change)
    df.drop_duplicates(subset = ['text'], inplace= True)

    # Remove mentions
    df['filtered_text'] = df.loc[:, 'text'].apply(lambda x: re.sub('@[A-Za-z0-9_]+','', x))

    # Remove hashtags
    df['filtered_text'] = df.loc[:, 'filtered_text'].apply(lambda x: re.sub('#[A-Za-z0-9_]+','', x))

    # Remove urls
    df['filtered_text'] = df.loc[:, 'filtered_text'].apply(lambda x: re.sub(r'https?://\S+', '', x))

    # Remove unusual characters (this line removes everything except alphanumeric characters and common punctuation or emoticon characters)
    df['filtered_text'] = df.loc[:, 'filtered_text'].apply(lambda x: re.sub(r'[^A-Za-z0-9 - ! ? , . ( ) & ^ -]', '', x))

    # Remove multiple spaces
    df['filtered_text'] = df.loc[:, 'filtered_text'].apply(lambda x: re.sub(r'\s+', ' ', x, flags=re.I))

    # Assess word counts for tweets (to show VADER effectiveness)
    df['word_count'] = df.loc[:, 'filtered_text'].apply(lambda x: len(str(x).split(' ')))

    # Return dataframe
    return df

# %%
def get_author_hist(df, max):
    # Get unique author tweet counts
    fig, ax = plt.subplots(1, 2, figsize = (12, 6))
    df.author_id.value_counts().hist(ax = ax[0], bins= range(0, 50, 5))
    ax[0].set_xticks(range(0, 55, 5))
    ax[0].set_xlim(1, 50)
    ax[0].set_title('Author Tweet Frequency Distribution')
    ax[0].set_ylabel('Frequency')
    ax[0].set_xlabel('Tweet count')

    df.author_id.value_counts().hist(ax = ax[1], bins= range(0, 50))
    ax[1].set_xticks(range(0, max))
    ax[1].set_xlim(1, max - 1)
    ax[1].set_title('Author Tweet Frequency Distribution (Zoomed in)')
    ax[1].set_ylabel('Frequency')
    ax[1].set_xlabel('Tweet count')

# %%
def run_vader(df):
    # Generate VADER sentiment analyzer object
    sid = SentimentIntensityAnalyzer()
    
    # On raw text
    # Perform calculation, see factors
    df['sentiment_compound_polarity']=df.text.apply(lambda x:sid.polarity_scores(x)['compound'])
    df['sentiment_neutral']=df.text.apply(lambda x:sid.polarity_scores(x)['neu'])
    df['sentiment_negative']=df.text.apply(lambda x:sid.polarity_scores(x)['neg'])
    df['sentiment_pos']=df.text.apply(lambda x:sid.polarity_scores(x)['pos'])

    # Create classification
    df['sentiment_type']=''
    df.loc[df.sentiment_compound_polarity>0,'sentiment_type']='POSITIVE'
    df.loc[df.sentiment_compound_polarity==0,'sentiment_type']='NEUTRAL'
    df.loc[df.sentiment_compound_polarity<0,'sentiment_type']='NEGATIVE'

    # On processed text
    # Perform calculation, see factors
    df['sentiment_compound_polarity_f']=df.filtered_text.apply(lambda x:sid.polarity_scores(x)['compound'])
    df['sentiment_neutral_f']=df.filtered_text.apply(lambda x:sid.polarity_scores(x)['neu'])
    df['sentiment_negative_f']=df.filtered_text.apply(lambda x:sid.polarity_scores(x)['neg'])
    df['sentiment_pos_f']=df.filtered_text.apply(lambda x:sid.polarity_scores(x)['pos'])

    # Create classification
    df['sentiment_type_f']=''
    df.loc[df.sentiment_compound_polarity_f>0,'sentiment_type_f']='POSITIVE'
    df.loc[df.sentiment_compound_polarity_f==0,'sentiment_type_f']='NEUTRAL'
    df.loc[df.sentiment_compound_polarity_f<0,'sentiment_type_f']='NEGATIVE'

    # Return dataframe
    return df

# %%
def mismatches(df):   
    # Find mismatches
    mismatches = df[df['sentiment_type'] != df['sentiment_type_f']]

    # See summary table to determine directional shift
    count = pd.DataFrame()
    count['raw'] = mismatches.sentiment_type.value_counts()
    count['filtered'] = mismatches.sentiment_type_f.value_counts()
    count['filtered_change'] = count['filtered'] - count['raw'] 

    return count

# %%
# Get distribution of tweet sentiment
def get_dist(sentiment_col_name, game_name):
    # Build plot
    plt.subplots(figsize = (18, 6))
    plt.title('Distribution of Sentiment on In-Game Tweets - ' + game_name)
    sns.kdeplot(sentiment_col_name, color='blue')
    plt.xlabel('Sentiment Score')
    plt.show()

    # Conduct shapiro test
    return shapiro(sentiment_col_name)

# %%
def create_word_cloud(pos_col, neg_col, game, extreme = False, stop = []):
    # Remove stopwords
    stopper = stopwords.words('english')
    pos_col = pos_col.apply(lambda x: ' '.join(x for x in x.split() if x not in stopper))
    neg_col = neg_col.apply(lambda x: ' '.join(x for x in x.split() if x not in stopper))

    # Remove punctuation, transform to lowercase
    pos_col = pos_col.apply(lambda x: re.sub(r'[^\w\s]','', x))
    pos_col = pos_col.str.lower()

    neg_col = neg_col.apply(lambda x: re.sub(r'[^\w\s]','', x))
    neg_col = neg_col.str.lower()

    # Create single word string to feed into viz
    pos = ' '.join(pos_col)
    neg = ' '.join(neg_col)

    # Add some custom stopwords if needed through word cloud
    stop_words = STOPWORDS.update(stop)

    pos_words = WordCloud(width=600,height=400, stopwords = stop_words, collocations=False, background_color='white').generate(pos)
    neg_words = WordCloud(width=600,height=400, stopwords = stop_words, collocations=False, background_color='white').generate(neg)
    
    # Create viz
    plt.subplots(figsize = (18, 6))
    plt.subplot(1,2,1)

    # Render title
    if(extreme == True):
        title_pos = 'Common Words from Extreme Positive Tweets: ' + game
        title_neg = 'Common Words from Extreme Negative Tweets: ' + game
    else:
        title_pos = 'Common Words from Positive Tweets: ' + game
        title_neg = 'Common Words from Negative Tweets: ' + game
            
    # Display
    plt.title(title_pos)
    plt.imshow(pos_words)
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.title(title_neg)
    plt.imshow(neg_words)
    plt.axis('off')
    plt.show();

# %%
# Get distribution of tweets
def get_dist_vol(col, game):
    # Build plot
    plt.subplots(figsize = (18, 6))
    plt.title('Distribution of Tweet Volume per Minute - ' + game)
    sns.kdeplot(col, color='blue')
    plt.xlabel('Tweet volume')
    plt.show()

    # Conduct shapiro
    return shapiro(col)


