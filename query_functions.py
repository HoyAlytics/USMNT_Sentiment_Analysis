# %%
import pandas as pd
import requests
import json
import time
import datetime

def twitter_query(username, hashtag, start_date, end_date):
    authenticator = {'Authorization' : "Bearer AAAAAAAAAAAAAAAAAAAAAJ2%2FhAEAAAAAGKgmH%2Bx6OUBv7AucQ%2FBs2uik%2Fso%3D8ZHrJyLIuQVQshfx3R5M5Sfp5FChMxt1Crak6HVJKQf0Yphj3f"}
    search_url = "https://api.twitter.com/2/tweets/search/recent"
    search_query = "(@" + username + " OR to:" + username + " OR #" + hashtag + ") -is:retweet lang:en"
    
    #start_date = (datetime.datetime.strptime(start_date,'%Y-%m-%d').isoformat("T") + "Z")
    #end_date = (datetime.datetime.strptime(end_date,'%Y-%m-%d').isoformat("T") + "Z")
    user_fields = ['description,id,location,name,profile_image_url,protected,public_metrics,username,verified']
    tweet_fields = ['attachments,author_id,conversation_id,created_at,geo,id,in_reply_to_user_id,lang,public_metrics,possibly_sensitive,referenced_tweets,reply_settings,source,text']
    
    

    global all_tweets_df
    all_tweets_df = pd.DataFrame()
    next_token = None
    
    while True:
        query_params = {
            'query' : search_query,
            'start_time' : start_date,
            'end_time' : end_date,
            'max_results' : 100,
            'expansions' : ['author_id,attachments.media_keys'],  
            'user.fields' : user_fields,
            'tweet.fields' : tweet_fields,
            'next_token' : next_token
        }

        query_results = requests.get(search_url, query_params, headers = authenticator)
        page_tweets_df = pd.DataFrame.from_dict(query_results.json()['data'])
        page_tweets_df = pd.merge(left = page_tweets_df, right = pd.DataFrame.from_dict(query_results.json()['includes']['users']).rename(columns = {'id':'author_id', 'public_metrics':'user_metrics'}), on = "author_id", suffixes= ["_tweet", "_user"])

        all_tweets_df = pd.concat([all_tweets_df, page_tweets_df], ignore_index = True)

        try:
            next_token = query_results.json()["meta"]["next_token"]
        except KeyError:
            break

        time.sleep(3)



# %%
