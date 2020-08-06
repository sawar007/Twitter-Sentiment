#!/usr/bin/env python
# coding: utf-8

# In[1]:


from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


# In[2]:


plt.style.use('fivethirtyeight')


# In[3]:


consumer_key = "8tcSQzvzZLRA0ltJZ3XE4siMy"
consumer_secret = "wyiLbp1hQ5gYzANlPnlpr1SHDsK9ry8OquO1U348kWJQCBkyEp"

# Access token and secret given by twitter with API
access_token = "3163679389-Cf4OiF6Qs2UFfeXg7ACwSkJMNEircIfUrhJXUKg"
access_secret = "7Eld3oQpXgCMA01SBoTGJBlsJLhLhnoJ7SryXn7irGHVX"


# In[4]:


import tweepy


# In[5]:


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)


# In[6]:


api = tweepy.API(auth)


# In[90]:


# posts = api.user_timeline(screen_name = 'bjp', count=100, language ='en', tweet_mode ='extended')
posts = api.search('java',count=10000,language='english')


# In[91]:


df = pd.DataFrame([ post.text for post in posts ] , columns=['Tweets'])
df


# In[92]:


def getpolarity(text):
    return TextBlob(text).sentiment.polarity
def getsubjectivity(text):
    return TextBlob(text).sentiment.subjectivity
def cleantext(text):
    text = re.sub(r'@[A-Za-z0-9]+','',text)
    text = re.sub(r'#','',text)
    text = re.sub(r'RT[\s]+','',text)
    text = re.sub(r'https?:\/\/\S+','',text)
    return text


# In[93]:


df['Tweets']=df['Tweets'].apply(cleantext)
df['Polarity'] = df['Tweets'].apply(getpolarity)
df['Subjectivity'] = df['Tweets'].apply(getsubjectivity)
df


# In[94]:


allwords = ''.join([twts for twts in df['Tweets'] ] )
wordCloud = WordCloud(width = 500, height = 300, random_state = 21, max_font_size= 119).generate(allwords)
plt.imshow(wordCloud, interpolation ='bilinear')
plt.axis('off')


# In[95]:


def score(point):
    if point<0:
        return 'Negative'
    if point ==0:
        return 'Neutral'
    else:
        return 'Positive'


# In[96]:


df["Analysis"] = df['Polarity'].apply(score)
df


# In[97]:


plt.figure(figsize =(8,6))
for i in range(df.shape[0]):
    plt.scatter(df['Polarity'][i],df['Subjectivity'][i],color='Blue')
plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()


# In[98]:


ptweets = df[df.Analysis=='Positive']
positive_percentage = (ptweets.shape[0]/df.shape[0])*100
positive_percentage


# In[99]:


ntweets = df[df.Analysis =='Negative']
negative_percentage = (ntweets.shape[0]/df.shape[0])*100
negative_percentage


# In[100]:


nutweets = df[df.Analysis =='Neutral']
neutral_percentage = (nutweets.shape[0]/df.shape[0])*100
neutral_percentage


# In[101]:


df['Analysis'].value_counts()


# In[102]:


plt.title('Sentiment Analysis')
plt.xlabel('Sentiments')
plt.ylabel('Counts')
df['Analysis'].value_counts().plot(kind='pie')


# In[ ]:




