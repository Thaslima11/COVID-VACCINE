# -*- coding: utf-8 -*-

# -- Sheet --
import os
import numpy as np
import plotly.express as px
import neattext as ntx
import pandas as pd
import re
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from textblob import TextBlob
from wordcloud import WordCloud
import random
from collections import Counter
from operator import itemgetter


print("Data Analysis of Covid Vaccines")

# Displaying the sample first ten rows in CSV
country_vaccine_dataframe = pd.read_csv(r'/Users/thaslimanasreen/Downloads/country_vaccinations.csv')
country_vaccine_dataframe.head(n=10)

# Removing the row if any of the column has NULL values
country_vaccine_dataframe.dropna(subset=country_vaccine_dataframe.columns,
                                 inplace=True)
# Displaying the dataframe without Null values
country_vaccine_dataframe.head(n=10)


# The function to get the vaccines available from the dataframes
def get_vaccines_available(cdf: pd.DataFrame):
    vaccines = cdf['vaccines'].unique()
    distinct_vaccines_available = []

    for vaccine in vaccines:
        for v in vaccine.split(','):
            v = v.strip()
            if v not in distinct_vaccines_available:
                distinct_vaccines_available.append(v)
    return distinct_vaccines_available


vaccines_list = get_vaccines_available(country_vaccine_dataframe)
print("Vaccines that are currently being used in the market:")
print(', '.join(vaccines_list))
# creating the below dictionaries to create a subsets from the CSV file
vaccines_country = {}  # which vaccines are used by different countries
people_vaccinated = {}  # number of people vaccinated
vaccinations_percentage = {}  # vaccination percentage
countries_list = country_vaccine_dataframe['country'].unique()

for vaccine in vaccines_list:
    vaccines_country[vaccine] = []

for country in countries_list:
    country_dataframe = country_vaccine_dataframe.loc[country_vaccine_dataframe['country'] == country]
    country_vaccinations = country_dataframe['people_vaccinated'].sum()
    people_vaccinated[country] = country_vaccinations
    c_vaccines = country_dataframe['vaccines'].unique()

    country_dataframe = country_dataframe.dropna(subset=['people_vaccinated_per_hundred'])
    if not country_dataframe.empty:
        vaccinations_percentage[country] = list(country_dataframe['people_vaccinated_per_hundred'])[-1]
    else:
        vaccinations_percentage[country] = 0

    for vaccine in c_vaccines:
        for v in vaccine.split(','):
            v = v.strip()
            vaccines_country[v].append(country)

for vaccine in vaccines_country:
    print(f"Vaccine: {vaccine}:")
    print(f"Number of countries using the vaccines: {len(vaccines_country[vaccine])}")
    print(f"List of countries that are using it: \n{vaccines_country[vaccine]}")
    print(f"-" * 200, "\n")

# prining the created subsets
print('The Count of People vaccinated across countries\n')
print(people_vaccinated)
print('The vaccine percentage in Countries\n')
print(vaccinations_percentage)
print('The vaccines used by the list of countries\n')
print(vaccines_country)
print(type(vaccines_country))

# defining N and N1 to be used in lowest and highest vaccines.
N = 10
N1 = 20
# N=10 largest values in dictionary
# sorted() + itemgetter() + items() to get the sorted list by vaccination rate

res = dict(sorted(people_vaccinated.items(), key=itemgetter(1), reverse=True)[:N])
lres = dict(sorted(people_vaccinated.items(), key=itemgetter(1))[:N1])
print('Countries with Lowest Vaccines Rate')
print(lres)

print('Top 10 Countries with Highes vaccination counties')
print(res)
print("Displaying the graph for Lowest vaccination rate countrywise")
# Displaying the graph for Lowest vaccinate rate
fig = plt.figure()
plt.bar(lres.keys(), lres.values(), color='r')
plt.xticks(rotation='vertical')
plt.title("Countries with Lowest Vaccination rate")
plt.show()

# Displaying the graph for highest vaccinate rate
print("The top 10 countries with highest vaccination rate  " + str(res))
l1 = list(res.keys())
l2 = list(res.values())
plt.barh(l1, l2, color='g')
plt.title("Top 10 Countries with Highest Vaccination rate")
plt.show()

vaccine_counts = {}
for v in vaccines_country:
    vaccine_counts[v] = len(vaccines_country[v])

# creating a list to plot the pie chart inorder to show usage percentage of all the available vaccines
newlist1 = list()  # to store the dictionary keys which is vaccine name
newlist2 = list()  # to store the dictionary values which is count of countries using the vaccines
for a in vaccine_counts.keys():
    newlist1.append(a)
for b in vaccine_counts.values():
    newlist2.append(b)
print('list')
print(newlist1)
print('count')
print(vaccine_counts.keys())
print(vaccine_counts.values())
x = np.char.array(newlist1)
y = np.array(newlist2)

# displaying the vaccination usage
print("Display Vaccine Usage Across Countries")
colors = ['yellowgreen', 'red', 'gold', 'lightskyblue', 'white', 'lightcoral', 'blue', 'pink', 'darkgreen', 'yellow',
          'grey', 'violet', 'magenta', 'cyan']
porcent = 100. * y / y.sum()

patches, texts = plt.pie(y, colors=colors, startangle=90, radius=1.2)
labels = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(x, porcent)]

sort_legend = True
if sort_legend:
    patches, labels, dummy = zip(*sorted(zip(patches, labels, y),
                                         key=lambda x: x[2],
                                         reverse=True))

plt.legend(patches, labels, loc='center left', bbox_to_anchor=(-0.1, 1.),
           fontsize=8)
plt.savefig('piechart.png', bbox_inches='tight')
plt.title("Vaccine Usage Across Countries")
plt.show()

# Country with highest vaccination rate
max_v_country = max(people_vaccinated, key=people_vaccinated.get)
print("The name of Country with highest vaccinations: ", max_v_country)

# In the country with the highest vaccination rate, getting the number of people vaccinated in the country over a time
max_vaccina_country_dataf = country_vaccine_dataframe.loc[country_vaccine_dataframe['country'] == max_v_country]
max_vaccina_country_dataf = max_vaccina_country_dataf.sort_values(by='date')
print("The name of country which vaccinated most of its population:", max_vaccina_country_dataf)
figure = px.line(max_vaccina_country_dataf, x='date', y='people_vaccinated',
                 title='Number of People Vaccinated in 100 millions (India)', color_discrete_sequence=['green'])
figure.show()

# The country which vaccinated a largest percentage of its population
max_per_country_population = max(vaccinations_percentage, key=vaccinations_percentage.get)
print("Country which vaccinated larger percentage of its population: ", max_per_country_population)
print("Percentage of vaccinated people: ", vaccinations_percentage[max_per_country_population])

# The number of people vaccinated in the country over a time for the country who vaccinated large number of its population
max_per_cntry_dataframe = country_vaccine_dataframe.loc[
    country_vaccine_dataframe['country'] == max_per_country_population]
max_per_cntry_dataframe = max_per_cntry_dataframe.sort_values(by='date')
figure = px.line(max_per_cntry_dataframe, x='date', y='people_vaccinated',
                 title='Number of People Vaccinated in Gibraltar over a time', color_discrete_sequence=['green'])
figure.show()

print("Sentiment Analysis of Covid Vaccines")

data = pd.read_csv(r'/Users/thaslimanasreen/Downloads/vaccination_all_tweets.csv')
print("The total number of tweet in the dataset is " + str(data.shape[0]))
print("The summarization of Data contained in CSV")
print(data.info())
print("Printing the first 5 Tweets\n")
print(data.head())
print("The number of missing data in the each columns")
data.isna().sum()

#removing the duplicate tweets

data=data.drop_duplicates('text')
print("Count of tweets after removing duplicates")
print(data.shape)

data['date'] = pd.to_datetime(data['date']).dt.date  # converting date column to date format
print("Displaying dataframe after conversion")
print(data.head())

print("The top 10 location wise tweets about vaccine")
plt.figure(figsize=(15, 10))
data['user_location'].value_counts().nlargest(10).plot(kind='pie')
plt.title('The top 10 location wise tweets about vaccine')
plt.xticks(rotation=60)



number_of_days = len(data['date'].unique())
print(type(number_of_days))
print('The total number of days considered', number_of_days)
print("Retrieving the 2 most old tweets about Covid Vaccine")  # Number of days considered
print(data.sort_values(by=['date'], ascending=[True]).head(2))
data.drop(columns={"id", "user_name", "user_description", "user_created", "user_followers", \
                   "user_friends", "user_favourites", "user_verified", "hashtags", "source", "retweets", "favorites",
                   "is_retweet"}, inplace=True)

pd.set_option('display.max_colwidth', 700)
print(data.head())
#Data has been cleaned using neattext library
data['clean_data'] = data['text'].apply(ntx.remove_hashtags)
data['clean_data'] = data['clean_data'].apply(ntx.remove_urls)
data['clean_data'] = data['clean_data'].apply(ntx.remove_multiple_spaces)
data['clean_data'] = data['clean_data'].apply(ntx.remove_special_characters)
data['clean_data'] = data['clean_data'].apply(ntx.remove_userhandles)
data[['clean_data', 'text']].head()

# NLTK has been used to remove stop words inorder to assign subjectivity and polarity to the tweets by TextBlob, so imported the neccessary libraries


lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
nltk.download('wordnet')
# The Polarity of the tweet will not change because of stop words so removing the stopwords
stop_words = stopwords.words('english')
len(stop_words), stop_words[5:10]


# The below function is used to remove the stop words from tweet
def Stop_word_removal(tweet):
    tweet_without_stopW = tweet
    tweet_without_stopW = " ".join(x for x in tweet_without_stopW.split() if x not in stop_words)
    return tweet_without_stopW


data['clean_data'] = data['clean_data'].apply(lambda x: Stop_word_removal(x))
print("The Original Tweet befor removing stopword")
print(data.head(2))
print("Printing the Tweet without stopword")
print(data['clean_data'])


# The below function is used to assign subjectivity and polarity to the tweet.
def textblob_func(text):
    sentimt = TextBlob(text)
    tweet_polarity = sentimt.sentiment.polarity
    tweet_subjectivity = sentimt.sentiment.subjectivity

    if tweet_polarity < 0:
        resp = 'Negative'

    elif tweet_polarity > 0:
        resp = 'Positive'

    elif tweet_polarity == 0:
        resp = "Neutral"

    result = {'polarity': tweet_polarity, 'subjectivity': tweet_subjectivity, 'sentiment': resp}

    return result


print("printing data['clean_data'][5]")
print(data['clean_data'][5])
print("Printing the sample result of textblob_function for the tweet")
print(textblob_func(data['clean_data'][5]))

#Appling the textblob_fun on the cleaned tweet
data['results'] = data['clean_data'].apply(textblob_func)
#Printing the polarity/subjectivity/neutral for the cleaned tweet
data.drop(columns={"user_location", 'text'}, inplace=True)
print("The sample result of polarity/subjectivity/neutral for the cleaned tweeet")
print(data.head(2))
data = data.join(pd.json_normalize(data=data['results']))
print("Displaying the tweets with the corresponding Polarity, Subjectivity and Sentiment")
print(data.head())

# segregating the tweets
positiveTweet = data[data['sentiment'] == 'Positive']['clean_data']
negativeTweet = data[data['sentiment'] == 'Negative']['clean_data']
neutralTweet = data[data['sentiment'] == 'Neutral']['clean_data']


#These colour functions will be called based on the sentiment of tweet
def red_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(10, 100%%, %d%%)" % random.randint(40, 100)


def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)


def green_color_func(word, font_size, position, orientation, random_state=None,
                     **kwargs):
    return "hsl(60, 100%%, %d%%)" % random.randint(60, 100)
# The below function is used to create wordclouds


def word_cloud(tweet_cat, title):
    forcloud = ' '.join([tweet for tweet in tweet_cat])
    wordcloud = WordCloud(width=500, height=300, random_state=5, max_font_size=110).generate(forcloud)
    plt.imshow(wordcloud, interpolation='bilinear')
    if (title == 'Positive Tweets'):
        plt.imshow(wordcloud.recolor(color_func=green_color_func, random_state=3),
                   interpolation="bilinear")
    elif (title == 'Negative Tweets'):
        plt.imshow(wordcloud.recolor(color_func=red_color_func, random_state=3),
                   interpolation="bilinear")
    else:
        plt.imshow(wordcloud.recolor(color_func=grey_color_func, random_state=3),
                   interpolation="bilinear")

    plt.title(title)
    plt.axis('off')
    plt.show()


plt.figure(figsize=(10, 8))
# To create Positive, Negative and Neutral Tweets
print("Displaying the word cloud of positive tweets")
word_cloud(positiveTweet, 'Positive Tweets')
print("Displaying the word cloud of negative tweets")
word_cloud(negativeTweet, 'Negative Tweets')
print("Displaying the word cloud of neutral tweets")
word_cloud(neutralTweet, 'Neutral Tweets')

pos_token = [token for line in positiveTweet for token in line.split()]
neg_token = [token for line in negativeTweet for token in line.split()]
neu_tokens = [token for line in neutralTweet for token in line.split()]




# to get the most used words in the vaccine related tweets
def get_most_used_token(tweets, num=30):
    print("\nInside get_most_used_token\n")
    word_tok = Counter(tweets)
    most_common_tokens = word_tok.most_common(num)
    return dict(most_common_tokens)


def token_df_visual(x, title):
    print("Inside Token_visual_func\n")
    df = pd.DataFrame(get_most_used_token(x).items(), columns=['words', 'count'])
    if(title=='Positive Tweet Tokens'):
        tk = px.bar(df, x='words', y='count', title=title,color_discrete_sequence = ['green'])
        tk.show()
    elif(title=='Negative Tweet Tokens'):
        tk = px.bar(df, x='words', y='count', title=title,color_discrete_sequence = ['red'])
        tk.show()
    else:
        tk = px.bar(df, x='words', y='count', title=title,color_discrete_sequence = ['grey'])
        tk.show()

print("Displaying the tokens of positive tweets")
token_df_visual(pos_token, 'Positive Tweet Tokens')
print("Displaying the tokens of negative tweets")
token_df_visual(neg_token, 'Negative Tweet Tokens')
print("Displaying the tokens of neutral tweets")
token_df_visual(neu_tokens, 'Neutral Tweet Tokens')

print("The polarity and Subjectivity of Tweet")
pol_sub = px.scatter(data, x='polarity', y='subjectivity',marginal_x="histogram", marginal_y="rug",title='The polarity and Subjectivity of Tweets',color_discrete_sequence = ['magenta'])
pol_sub.show()


def percent(x, y):
    return round(len(x) / data.shape[0] * 100, 3)

print("percentage")
print(percent(positiveTweet, 'positive'))
print(percent(negativeTweet, 'negative'))
print(percent(neutralTweet, 'neutral'))
print("pie chart")
senti_tweets = ['Positive', 'Negative', 'Neutral']
perc=[]
print("Displaying the sentiment of tweets in percentage")
for i in senti_tweets:
    if(i=='Positive'):
        a=percent(positiveTweet, i)
    elif(i=='Negative'):
        a = percent(negativeTweet, i)
    else:
        a=percent(neutralTweet, i)
    x=str(a)+"%"
    print("Percentage of " + i + " tweets :",x)
    perc.append(a)
explode = (0, 0.1, 0, 0)
print("pie chart per")
print(perc)
fig1, ax1 = plt.subplots()
ax1.pie(perc, labels=senti_tweets, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.title('Sentiment of the tweets')

plt.show()
print("Percentage")
print(perc)

data['sentiment'].value_counts().plot(kind='bar')
data.columns

deep = data.drop(columns="results")
deep.head(2)

# References created for the 5 Vaccines that are used mostly in the market
pfizer_reference = ["Pfizer", "pfizer", "pfizer-bioNtech", "Pfizer–BioNTech", "BioNTech", "biontech"]
bbiotech_reference = ["covax", "covaxin", "Covax", "Covaxin", "Bharat Biotech", "bharat biotech", "BharatBiotech",
                      "bharatbiotech"]
sputnik_reference = ["russia", "sputnik", "Sputnik", "V"]
astra_refeference = ['sii', 'SII', 'adar poonawalla', 'Covishield', 'covishield', 'astra', 'zenca',
                     'Oxford–AstraZeneca', 'astrazenca', 'oxford-astrazenca', 'serum institiute']
moderna_reference = ['moderna', 'Moderna', 'mRNA-1273', 'Spikevax']


def refer_func(tweet, refs):
    flag = 0
    for ref in refs:
        if tweet.find(ref) != -1:
            flag = 1
    return flag


deep['pfizer'] = deep['clean_data'].apply(lambda x: refer_func(x, pfizer_reference))
deep['bbiotech'] = deep['clean_data'].apply(lambda x: refer_func(x, bbiotech_reference))
deep['sputnik'] = deep['clean_data'].apply(lambda x: refer_func(x, sputnik_reference))
deep['astra'] = deep['clean_data'].apply(lambda x: refer_func(x, astra_refeference))
deep['moderna'] = deep['clean_data'].apply(lambda x: refer_func(x, moderna_reference))
print(deep.pfizer.value_counts(), deep.bbiotech.value_counts(), deep.sputnik.value_counts(), deep.astra.value_counts(),
 deep.moderna.value_counts())
print("The sample tweet and Polarity/Subjectiviy of Biotech")
print("The sample tweet and Polarity/Subjectiviy of Pfizer")
deep[deep['bbiotech'] == 1].head()
deep[deep['pfizer'] == 1].head()


def stats_vaccine_func(p, b, s, a, m):
    for i in p, b, s, a, m:
        print("In stats_vaccine_func")
        print("Printing the polarity and subjectivity and other statistics")
        print(deep[deep[i] == 1][[i, 'polarity', 'subjectivity']].groupby(i).agg([np.mean, np.max, np.min, np.median]))


stats_vaccine_func('pfizer', 'bbiotech', 'sputnik', 'astra', 'moderna')

pfizer = deep[deep['pfizer'] == 1][['date', 'polarity']]
bbiotech = deep[deep['bbiotech'] == 1][['date', 'polarity']]
sputnik = deep[deep['sputnik'] == 1][['date', 'polarity']]
astra = deep[deep['astra'] == 1][['date', 'polarity']]
moderna = deep[deep['moderna'] == 1][['date', 'polarity']]

pfizer = pfizer.sort_values(by='date', ascending=True)
bbiotech = bbiotech.sort_values(by='date', ascending=True)
sputnik = sputnik.sort_values(by='date', ascending=True)
astra = astra.sort_values(by='date', ascending=True)
moderna = moderna.sort_values(by='date', ascending=True)

pfizer['Avg Polarity'] = pfizer.polarity.rolling(20, min_periods=3).mean()
bbiotech['Avg Polarity'] = bbiotech.polarity.rolling(20, min_periods=3).mean()
sputnik['Avg Polarity'] = sputnik.polarity.rolling(20, min_periods=3).mean()
astra['Avg Polarity'] = astra.polarity.rolling(5, min_periods=3).mean()
moderna['Avg Polarity'] = moderna.polarity.rolling(20, min_periods=3).mean()
bbiotech.head(10)

p, b, s, a, m = pfizer, bbiotech, sputnik, astra, moderna
print("printing")
print(pfizer)
print(type(pfizer))

ax = plt.gca()
print("Displaying the Pfizer vaccine polarity vs Time")
pfz = px.scatter(p, x="date", y="Avg Polarity", marginal_x="histogram", marginal_y="rug",title='Pfizer Vaccine', color_discrete_sequence = ['green'])
pfz.show()
print("Displaying the Biotech vaccine polarity vs Time")
pfz = px.scatter(b, x="date", y="Avg Polarity", marginal_x="histogram", marginal_y="rug",title='Bharat Biotech Vaccine', color_discrete_sequence = ['orange'])
pfz.show()
print("Displaying the Sputnik vaccine polarity vs Time")
pfz = px.scatter(s, x="date", y="Avg Polarity", marginal_x="histogram", marginal_y="rug",title='Sputnik Vaccine', color_discrete_sequence = ['red'])
pfz.show()
print("Displaying the Astrazenaca vaccine polarity vs Time")
pfz = px.scatter(a, x="date", y="Avg Polarity", marginal_x="histogram", marginal_y="rug",title='AstraZence/Covishield Vaccine', color_discrete_sequence = ['blue'])
pfz.show()
print("Displaying the Moderna vaccine polarity vs Time")
pfz = px.scatter(m, x="date", y="Avg Polarity", marginal_x="histogram", marginal_y="rug",title='Moderna Vaccine', color_discrete_sequence = ['magenta'])
pfz.show()
#a =px.line(a, x="date", y="Avg Polarity", title='Pfizer', color_discrete_sequence = ['red'])
#a.show()
overall=pd.DataFrame()
overall['date'] = sorted(deep['date'].unique())
senti_v=list()
#The overall sentiment of the public about the vaccines
for date in overall['date']:
    senti_v.append(deep[deep['date']==date].polarity.mean())
overall['Sentiment']=senti_v
print("Displaying overall sentiment of the public about the vaccines")
fig = px.bar(overall, x="date", y="Sentiment", title='Overall Sentiment of the public tweets about Covid Vaccines', color_discrete_sequence = ['green'])
fig.show()
print("The End Data")


