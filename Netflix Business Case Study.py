#!/usr/bin/env python
# coding: utf-8

# # Business Problem
# Analyze the data and generate insights that could help Netflix in deciding which type of shows/movies to produce and how they can grow the business in different countries.
# 

# # Netflix Case study Dataset 

# # Objectives of the Case study

# * Perform EDA on the given dataset and find insights.
# * Provide Useful Insights and Business recommendations that can help the business to grow.
# 
# 
# 
# 
# 

# 
# 
# 
# 
# 

# # 1. Importing Libraries , Loading the data and Basic Observations

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('Netflix_dataset.csv')


# In[3]:


df.head()


# These are the first 5 rows of the dataset. The actual size of the dataset is given below. total 8807 rows and 12 columns.

# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.nunique()


# These are total features of our dataset. It is seen that show_id column has all unique values, 
# Title column has all unique values i.e. total 8807 which equates with total rows in the dataset.
# Hence It can be concluded that ,
# 
# Total 8807 movies/TV shows data is provided in the dataset.

# In[7]:


df.describe()


# Only single column having numerical values. It gives idea of release year of the content ranges between what timeframe.
# Rest all the columns are having categorical data.

# In[8]:


df.describe(include = object)


# In[ ]:





# # 2. Data Cleaning

# Overall null values in each column of the dataset -

# In[9]:


df.isna().sum()


# * 3 missing values are found in duration column , and it is also found that by mistake those data got entered in rating column

# In[10]:


df[df['duration'].isna()]


# In[11]:


ind = df[df['duration'].isna()].index


# In[12]:


df.loc[ind] = df.loc[ind].fillna(method = 'ffill' , axis = 1)


# In[13]:


# replaced the wrong entries done in the rating column
df.loc[ind ,'rating'] = 'Not Available'


# In[14]:


df.loc[ind]


# In[ ]:





# * Fill the null values in rating column

# In[15]:


df[df.rating.isna()]


# In[16]:


indices = df[df.rating.isna()].index
indices


# In[18]:


df.loc[indices , 'rating'] = 'Not Available'


# In[17]:


df.loc[indices]


# In[ ]:





# In[18]:


df.rating.unique()


# In rating column , NR (Not rated) is same as UR (Unrated). lets change UR to NR.

# In[19]:


df.loc[df['rating'] == 'UR' , 'rating'] = 'NR'
df.rating.value_counts()


# In[ ]:





# * dropped the null from date_added column

# In[20]:


df.drop(df.loc[df['date_added'].isna()].index , axis = 0 , inplace = True)


# In[21]:


df['date_added'].value_counts()


# For 'date_added' column, all values confirm to date format, So we can convert its data type from object to datetime

# In[22]:


df['date_added'] = pd.to_datetime(df['date_added'])
df['date_added']


# In[ ]:





# We can add the new column 'year_added' by extracting the year from 'date_added' column

# In[23]:


df['year_added'] = df['date_added'].dt.year


# Similar way, We can add the new column 'month_added' by extracting the month from 'date_added' column

# In[24]:


df['month_added'] = df['date_added'].dt.month


# In[25]:


df[['date_added' , 'year_added' , 'month_added']].info()


# In[ ]:





# In[26]:


# total null values in each column
df.isna().sum()


# % Null values in each column

# In[27]:


round((df.isna().sum()/ df.shape[0])*100)


# We can see that, after cleaning some data we still have null values in 3 columns. These are much higher in numbers.
# 
# For some content - country is missing. (9%)
# 
# for some content - director names are missing (30%)
# 
# for some content - cast is missing (9%)

# In[ ]:





# # 3. Data Exploration and Non Graphical Analysis

# In[28]:


# 2 types of content present in dataset - either Movie or TV Show
df['type'].unique()


# In[29]:


movies  = df.loc[df['type'] == 'Movie']
tv_shows = df.loc[df['type'] == 'TV Show'] 


# In[30]:


movies.duration.value_counts()


# In[31]:


tv_shows.duration.value_counts()


# In[ ]:





# In[ ]:





# Since movie and TV shows both have different format for duration, we can change duration for movies as minutes & TV shows as seasons

# In[32]:


movies_dummy  = df.loc[df['type'] == 'Movie']
tv_shows_dummy = df.loc[df['type'] == 'TV Show'] 


# In[35]:


movies_dummy["duration"] = movies_dummy["duration"].str.split(" ")


# In[38]:


movies_dummy["duration"] = movies_dummy["duration"].apply(lambda x : x[0])


# In[40]:


movies_dummy.head()


# In[ ]:





# In[ ]:





# In[41]:


movies['duration'] = movies['duration'].str[:-3]
movies['duration'] = movies['duration'].astype('float')


# In[42]:


tv_shows['duration'] = tv_shows.duration.str[:-7].apply(lambda x : x.strip())
tv_shows['duration'] = tv_shows['duration'].astype('float')


# In[43]:


tv_shows.rename({'duration': 'duration_in_seasons'} ,axis = 1 , inplace = True)
movies.rename({'duration': 'duration_in_minutes'} ,axis = 1 , inplace = True)


# In[44]:


tv_shows.duration_in_seasons


# In[45]:


movies.duration_in_minutes


# In[ ]:





# when was first movie added on netflix and when is the most recent movie added on netflix as per data i.e. dataset duration

# In[46]:


timeperiod = pd.Series((df['date_added'].min().strftime('%C %B %Y') , df['date_added'].max().strftime('%C %B %Y')))
timeperiod.index = ['first' , 'Most Recent']
timeperiod


# In[ ]:





# The oldest and the most recent movie/TV show released on the Netflix in which year?

# In[43]:


df.release_year.min() , df.release_year.max() 


# In[44]:


df.loc[(df.release_year == df.release_year.min()) | (df.release_year == df.release_year.max())].sort_values('release_year')


# In[ ]:





# Which are different ratings available on Netflix in each type of content? Check the number of content released in each type.

# In[45]:


df.groupby(['type' , 'rating'])['show_id'].count()


# In[ ]:





# Working on the columns having maximum null values and the columns having comma separated multiple values for each record

# * Country column

# In[46]:


df['country'].value_counts()


# We see that many movies are produced in more than 1 country. Hence, the country column has comma separated values of countries.
# 
# This makes it difficult to analyse how many movies were produced in each country. We can use explode function in pandas to split the country column into different rows.
# 
# we are Creating a separate table for country , to avoid the duplicasy of records in our origional table after exploding.

# In[48]:


country_tb = df[['show_id' , 'type' , 'country']]
country_tb.dropna(inplace = True)
country_tb['country'] = country_tb['country'].apply(lambda x : x.split(','))
country_tb = country_tb.explode('country')
country_tb.head()


# In[51]:


# some duplicate values are found, which have unnecessary spaces. some empty strings found
country_tb['country'] = country_tb['country'].str.strip()


# In[52]:


country_tb.loc[country_tb['country'] == '']


# In[53]:


country_tb = country_tb.loc[country_tb['country'] != '']


# In[54]:


country_tb['country'].nunique()


# Netflix has movies from the total 122 countries.

# Total movies and tv shows in each country

# In[56]:


x = country_tb.groupby(['country' , 'type'])['show_id'].count().reset_index()
x.pivot(index = 'country' , columns = 'type' , values = 'show_id').sort_values('Movie',ascending = False)


# In[ ]:





# * Director column 

# In[54]:


df['director'].value_counts()


# There are some movies which are directed by multiple directors. Hence multiple names of directors are given in comma separated format.
# We will explode the director column as well. It will create many duplicate records in originaltable hence we created separate table for directors.

# In[57]:


dir_tb = df[['show_id' , 'type' , 'director']]
dir_tb.dropna(inplace = True)
dir_tb['director'] = dir_tb['director'].apply(lambda x : x.split(','))
dir_tb


# In[58]:


dir_tb = dir_tb.explode('director')


# In[59]:


dir_tb['director'] = dir_tb['director'].str.strip()


# In[58]:


# checking if empty stirngs are there in director column
dir_tb.director.apply(lambda x : True if len(x) == 0 else False).value_counts()


# In[61]:


dir_tb.head()


# In[62]:


dir_tb['director'].nunique()


# There are total 4993 unique directors in the dataset.

# Total movies and tv shows directed by each director

# In[64]:


x = dir_tb.groupby(['director' , 'type'])['show_id'].count().reset_index()
x.pivot(index= 'director' , columns = 'type' , values = 'show_id').sort_values('Movie' ,ascending = False)


# In[ ]:





# * 'listed_in' column to understand more about genres

# In[65]:


genre_tb = df[['show_id' , 'type', 'listed_in']]


# In[66]:


genre_tb['listed_in'] = genre_tb['listed_in'].apply(lambda x : x.split(','))
genre_tb = genre_tb.explode('listed_in')
genre_tb['listed_in'] = genre_tb['listed_in'].str.strip()


# In[67]:


genre_tb.head()


# In[68]:


genre_tb.listed_in.unique()


# In[69]:


genre_tb.listed_in.nunique()


# Total 42 genres present in dataset

# In[70]:


df.merge(genre_tb , on = 'show_id' ).groupby(['type_y'])['listed_in_y'].nunique()


# Movies have 20 genres and TV shows have 22 genres.

# In[71]:


# total movies/TV shows in each genre
x = genre_tb.groupby(['listed_in' , 'type'])['show_id'].count().reset_index()
x.pivot(index = 'listed_in' , columns = 'type' , values = 'show_id').sort_index()


# In[ ]:





# In[69]:


# Exploring cast column


# In[72]:


cast_tb = df[['show_id' , 'type' ,'cast']]
cast_tb.dropna(inplace = True)
cast_tb['cast'] = cast_tb['cast'].apply(lambda x : x.split(','))
cast_tb = cast_tb.explode('cast')
cast_tb


# In[73]:


cast_tb['cast'] = cast_tb['cast'].str.strip()


# In[74]:


# checking empty strings
cast_tb[cast_tb['cast'] == '']


# In[75]:


# Total actors on the Netflix
cast_tb.cast.nunique()


# In[76]:


# Total movies/TV shows by each actor
x = cast_tb.groupby(['cast' , 'type'])['show_id'].count().reset_index()
x.pivot(index = 'cast' , columns = 'type' , values = 'show_id').sort_values('TV Show' , ascending = False)


# #  4. Visual Analysis - Univariate & Bivariate

# * 4.1. Distribution of content across the different types

# In[121]:


types = df.type.value_counts()
plt.pie(types,  labels=types.index, autopct='%1.1f%%' , colors = ['blue' , 'orange'])
plt.title('Total_Movies and TV Shows')
plt.show()


# It is observed that , around 70% content is Movies and around 30% content is TV shows.

# * 4.2 Distribution of 'date_added' column

#  How has the number of movies/TV shows added on Netflix per year changed over the time?

# In[77]:


d = df.groupby(['year_added' ,'type' ])['show_id'].count().reset_index()
d.rename({'show_id' : 'total movies/TV shows'}, axis = 1 , inplace = True)


# In[78]:


plt.figure(figsize = (12,6))
sns.lineplot(data = d , x = 'year_added' , y = 'total movies/TV shows' , hue = 'type', marker = 'o'  , ms = 6)
plt.xlabel('year_added' , fontsize = 12)
plt.ylabel('total movies/TV shows' , fontsize = 12)
plt.title('total movies and TV shows by the year_added' , fontsize = 12)
plt.show()


# Observation: 
#    * The content added on the Netflix surged drastically after 2015.
#    * 2019 marks the highest number of movies and TV shows added on the Netflix.
#    * Year 2020 and 2021 has seen the drop in content added on Netflix, possibly because of Pandemic.
#     But still , TV shows content have not dropped as drastic as movies. In recent years TV shows are focussed more than Movies.
#     

# In[ ]:





# * 4.3 Distribution of 'Release_year' column

# How has the number of movies released per year changed over the last 20-30 years?

# In[79]:


d = df.groupby(['type' , 'release_year'])['show_id'].count().reset_index()
d.rename({'show_id' : 'total movies/TV shows'}, axis = 1 , inplace = True)
d


# In[83]:


plt.figure(figsize = (12,6))
sns.lineplot(data = d , x = 'release_year' , y = 'total movies/TV shows' , hue = 'type' , marker = 'o'  , ms = 6 )
plt.xlabel('release_year' , fontsize = 12)
plt.ylabel('total movies/TV shows' , fontsize = 12)
plt.title('total movies and TV shows by the release_year' , fontsize = 12)
plt.xlim( left = 2000 , right = 2021)
plt.xticks(np.arange(2000 , 2021 , 2))
plt.show()


# Observation: 
#    * 2018 marks the highest number of movie and TV show releases. 
#    * Since 2018, A drop in movies is seen and rise in TV shows is observed clearly, and TV shows surpasses the movies count in mid 2020.
#    * In recent years TV shows are focussed more than Movies.
#    * The yearly number of releases has surged drastically from 2015.

# * 4.4  Total movies/TV shows by each director

# In[81]:


# total Movies directed by top 10 directors
top_10_dir = dir_tb.director.value_counts().head(10).index
df_new = dir_tb.loc[dir_tb['director'].isin(top_10_dir)]


# In[82]:


plt.figure(figsize= (8 , 6))
sns.countplot(data = df_new , y = 'director' , order = top_10_dir , orient = 'v')
plt.xlabel('total_movies/TV shows' , fontsize = 12)
plt.xlabel('Movies/TV shows count')
plt.ylabel('Directors' , fontsize = 12)
plt.title('Total_movies/TVshows_by_director')
plt.show()


# Observation:
#    * The top 3 directors on Netflix in terms of count of movies directed by them are - Rajiv Chilaka, Jan Suter, Raúl Campos

# * 4.4  Checking Outliers for number of movies directed by each director

# In[93]:


x = dir_tb.director.value_counts()
x


# In[94]:


x.values


# In[87]:


def calculate_outliers(data):
    # Calculate the first quartile (Q1)
    q1 = np.percentile(data, 25)
    
    # Calculate the third quartile (Q3)
    q3 = np.percentile(data, 75)
    
    # Calculate the interquartile range (IQR)
    iqr = q3 - q1
    
    # Determine the lower and upper bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Identify outliers in the dataset
    outliers = [value for value in data.values if value < lower_bound or value > upper_bound]
    
    return outliers


def calculate_max_occurred_value(data):
    # Calculate the unique values and their counts in the dataset
    unique_values, value_counts = np.unique(data, return_counts=True)
    
    # Find the index of the maximum count
    max_count_index = np.argmax(value_counts)
    
    # Retrieve the corresponding unique value with the maximum count
    max_occurred_value = unique_values[max_count_index]
    
    return max_occurred_value


# In[98]:


unique_values, value_counts = np.unique(x, return_counts=True)


# In[99]:


unique_values


# In[100]:


value_counts


# In[101]:


z = np.unique(x, return_counts=True)
z


# In[95]:


outliers = calculate_outliers(x)  # Implement your outlier calculation method
#max_occurred_value = calculate_max_occurred_value(x)  # Implement your method to find the maximum-occurred value
set(outliers) 


# In[96]:


max_occurred_value = calculate_max_occurred_value(x)  # Implement your method to find the maximum-occurred value
max_occurred_value


# In[158]:


plt.figure(figsize = (12,6))
sns.boxplot(data=x, showfliers=True, whis=1.5 , orient = 'h')

# Calculate the outliers and maximum-occurred value
outliers = calculate_outliers(x)  # Implement your outlier calculation method
max_occurred_value = calculate_max_occurred_value(x)  # Implement your method to find the maximum-occurred value

# Annotate the plot
plt.text(0.95, 0.9, f"Outliers: {len(outliers)}", transform=plt.gca().transAxes, ha='right')
plt.text(0.95, 0.85, f"Max Occurred: {max_occurred_value}", transform=plt.gca().transAxes, ha='right')


plt.xlabel("Count of movies directed by each Director")
plt.xticks(np.arange(0,22,2))
plt.title("Boxplot with Outliers and Max Occurred Value")

# Show the plot
plt.show()


# It is Observed that maximum occured value is 1, which means maximum directors on the Netflix have directed 1 movie/Tv show. There are few directors who have directed more than 1 movies/tv shows and they are outliers.
# 

# In[ ]:





# * 4.5 Total movies/TV shows by each country

# In[159]:


# Lets check for top 10 countries
top_10_country = country_tb.country.value_counts().head(10).index
df_new = country_tb.loc[country_tb['country'].isin(top_10_country)]


# In[160]:


x = df_new.groupby(['country' , 'type'])['show_id'].count().reset_index()
x.pivot(index = 'country' , columns = 'type' , values = 'show_id').sort_values('Movie',ascending = False)


# In[161]:


plt.figure(figsize= (8,5))
sns.countplot(data = df_new , x = 'country' , order = top_10_country , hue = 'type')
plt.xticks(rotation = 90 , fontsize = 12)
plt.ylabel('total_movies/TV shows' , fontsize = 12)
plt.xlabel('')
plt.title('Total_movies/TVshows_by_country')
plt.show()


# In[244]:


top_10_country = country_tb.country.value_counts().head(10).index
country_tb['cat'] = country_tb['country'].apply(lambda x : x if x in top_10_country else 'Other Countries' )


# In[247]:


x = country_tb.cat.value_counts()

plt.figure(figsize = (8,8))
plt.pie(x , labels = x.index, autopct='%1.1f%%')
plt.title('Total Content produced in each country' , fontsize = 15)
plt.show()


# * Observation:
#     * United States is the HIGHEST contributor country on Netflix, followed by India and United Kingdom.
#     * Maximum content of Netflix which is around 75% , is coming from these top 10 countries.  Rest of the world only contributes 25% of the content.

# In[ ]:





# * 4.6 Total content distribution by release year of the content

# In[165]:


plt.figure(figsize= (12,6))
sns.boxplot(data = df , x = 'release_year')
plt.xlabel('release_year' , fontsize = 12)
plt.title('Total_movies/TVshows_by_release_year')
plt.xticks(np.arange(1940 , 2021 , 5))
plt.xlim((1940 , 2022))
plt.show()


# * Netflix have major content which is released in the year range 2000-2021
# * It seems that the content older than year 2000 is almost missing from the Netflix.

# * 4.7 Total movies/TV shows distribution by rating of the content

# In[200]:


m = movies.loc[~movies.rating.isin(['Not Available' , 'NC-17' , 'TV-Y7-FV'])]
m = m.rating.value_counts()
t = tv_shows.loc[~tv_shows.rating.isin(['Not Available' , 'R' , 'NR', 'TV-Y7-FV'])]
t = t.rating.value_counts()


fig, ax = plt.subplots(1,2, figsize=(14,8))
ax[0].pie(m , labels = m.index, autopct='%1.1f%%')
ax[0].set_title('Total_movies_by_rating')

ax[1].pie(t , labels = t.index, autopct='%1.1f%%')
ax[1].set_title('Total_TV_shows_by_rating')

plt.tight_layout()
plt.show()


# Highest number of movies and TV shows are rated TV-MA (for mature audiences), followed by TV-14 & R/TV-PG

# In[ ]:





# * 4.8 Total movies/TV shows distribution by duration of the content

# In[387]:


fig, ax = plt.subplots(2,1, figsize=(8,6))

sns.boxplot (data = movies , x = 'duration_in_minutes' ,ax =ax[0])
ax[0].set_xlabel('duration_in_minutes' ,  fontsize = 12)
ax[0].set_title('Total movies by duration')

sns.boxplot (data = tv_shows , x = 'duration_in_seasons' , ax = ax[1])
ax[1].set_xlabel('Number_of_seasons' ,  fontsize = 12)
ax[1].set_title('Total TV shows by duration')

plt.tight_layout()
plt.show()


# * Movie Duration: 50 mins - 150 mins is the range excluding potential outliers (values lying outside the whiskers of boxplot)
# * TV Show Duration: 1-3 seasons is the range for TV shows excluding potential outliers

# * 4.9  Total movies/TV shows in each Genre

# In[203]:


# Lets check the count for top 10 genres in Movies and TV_shows


# In[204]:


top_10_movie_genres = genre_tb[genre_tb['type'] == 'Movie'].listed_in.value_counts().head(10).index
df_movie = genre_tb.loc[genre_tb['listed_in'].isin(top_10_movie_genres)]


# In[205]:


top_10_TV_genres = genre_tb[genre_tb['type'] == 'TV Show'].listed_in.value_counts().head(10).index
df_tv = genre_tb.loc[genre_tb['listed_in'].isin(top_10_TV_genres)]


# In[206]:


plt.figure(figsize= (8,4))
sns.countplot(data = df_movie , x = 'listed_in' , order = top_10_movie_genres)
plt.xticks(rotation = 90 , fontsize = 12)
plt.ylabel('total_movies' , fontsize = 12)
plt.xlabel('Genres' , fontsize = 12)
plt.title('Total_movies_by_genre')
plt.show()


# In[209]:


plt.figure(figsize= (8,4))
sns.countplot(data = df_tv , x = 'listed_in' , order = top_10_TV_genres)
plt.xticks(rotation = 90 , fontsize = 12)
plt.ylabel('total_TV_Shows' , fontsize = 12)
plt.xlabel('Genres' , fontsize = 12)
plt.title('Total_TV_Shows_by_genre')
plt.show()


# * International Movies and TV Shows , Dramas , and Comedies are the top 3 genres on Netflix for both Movies and TV shows.

# # 5. Bivariate Analysis

# * 5.1  Lets check popular genres in top 20 countries

# In[278]:


top_20_country = country_tb.country.value_counts().head(20).index
top_20_country = country_tb.loc[country_tb['country'].isin(top_20_country)]


# In[279]:


x = top_20_country.merge(genre_tb , on = 'show_id').drop_duplicates()
country_genre = x.groupby([ 'country' , 'listed_in'])['show_id'].count().sort_values(ascending = False).reset_index()
country_genre = country_genre.pivot(index = 'listed_in' , columns = 'country' , values = 'show_id')


# In[372]:


plt.figure(figsize = (12,10))
sns.heatmap(data = country_genre , annot = True , fmt=".0f" , vmin = 20 , vmax = 250 )
plt.xlabel('Countries' , fontsize = 12)
plt.ylabel('Genres' , fontsize = 12)
plt.title('Countries V/s Genres' , fontsize = 12)


# Popular genres across countries: Action & Adventure, Children & Family Movies, Comedies, Dramas, International Movies & TV Shows, TV Dramas, Thrillers
# 
# Country-specific genres: Korean TV shows (Korea), British TV Shows (UK), Anime features and Anime series (Japan), Spanish TV Shows (Argentina, Mexico and Spain)
# 
# United States and UK have a good mix of almost all genres.
# 
# Maximum International movies are produced in India.

# In[ ]:





# 5.2 Country-wise Rating of Content

# In[352]:


x = top_20_country.merge(df , on = 'show_id').groupby(['country_x' , 'rating'])['show_id'].count().reset_index()


# In[358]:


country_rating = x.pivot(index = ['country_x'] , columns = 'rating' , values = 'show_id')


# In[374]:


plt.figure(figsize = (10,8))
sns.heatmap(data = country_rating , annot = True , fmt=".0f"  , vmin = 10 , vmax=200)
plt.ylabel('Countries' , fontsize = 12)
plt.xlabel('Rating' , fontsize = 12)
plt.title('Countries V/s Rating' , fontsize = 12)


# * Overall, Netflix has an large amount of adult content across all countries (TV-MA & TV-14).
# * India also has many titles rated TV-PG, other than TV-MA & TV-14.
# * Only US, Canada, UK, France and Japan have content for young audiences (TV-Y & TV-Y7).
# * There is scarce content for general audience (TV-G & G) across all countries except US.

# * 5.3  The top actors by country

# In[251]:


x = cast_tb.merge(country_tb , on = 'show_id').drop_duplicates()
x = x.groupby(['country' , 'cast'])['show_id'].count().reset_index()
x.loc[x['country'].isin(['United States'])].sort_values('show_id' , ascending = False).head(5)


# In[252]:


country_list = ['India'  , 'United Kingdom' , 'Canada' , 'France' , 'Japan']
top_5_actors = x.loc[x['country'].isin(['United States'])].sort_values('show_id' , ascending = False).head(5)


# In[253]:


for i in country_list:
    new = x.loc[x['country'].isin([i])].sort_values('show_id' , ascending = False).head(5)
    top_5_actors = pd.concat( [top_5_actors , new] , ignore_index = True)
    


# In[405]:


# top 5 actors in top countries and their movies/tv shows count
top_5_actors


# In[274]:


plt.figure(figsize = (10,10))
sns.barplot(data = top_5_actors , y = 'cast' , x = 'show_id' , hue = 'country')


# * 5.4 Top 5 directors by Genre

# In[493]:


genre_list = [ 'Children & Family Movies', 'Comedies','Dramas', 'International Movies', 'Documentaries' ,
              'International TV Shows', 'Sci-Fi & Fantasy', 'Thrillers', 'Horror Movies']

x = dir_tb.merge(genre_tb , on = 'show_id').groupby([ 'listed_in' , 'director',])['show_id'].count().reset_index()

top_5_dir = x.loc[x['listed_in'] == 'Action & Adventure'].sort_values('show_id' , ascending = False).head()

for i in genre_list:
    new = x.loc[x['listed_in'] == i].sort_values('show_id' , ascending = False).head()
    top_5_dir = pd.concat([top_5_dir , new])
    
top_5_dir


# * 5.5  Top 5 genres in each country

# In[342]:


x = genre_tb.merge(country_tb , on = 'show_id').drop_duplicates()
x = x.groupby(['country' , 'listed_in'])['show_id'].count().reset_index()
x.loc[x['country'] == 'United States'].sort_values('show_id' , ascending = False).head(5)

country_list = ['India'  , 'United Kingdom' , 'Canada' , 'France' , 'Japan']
top_5_genre = x.loc[x['country'].isin(['United States'])].sort_values('show_id' , ascending = False).head(5)

for i in country_list:
    new = x.loc[x['country'] == i].sort_values('show_id' , ascending = False).head(5)
    top_5_genre = pd.concat( [top_5_genre , new] , ignore_index = True)


# In[343]:


top_5_genre


# In[ ]:





# *  5.6  Variation in duration of movies by Release year

# In[286]:


plt.figure(figsize = (12,8))
sns.scatterplot(movies['duration_in_minutes'], movies['release_year'],  alpha=0.5)
plt.xlim((0,200))


# * Observation
#     * The movies shorter than 150 minutes duration have increased drastically after 2000 while movies longer than 150 minutes are not much popular.
#     * There is a huge surge in the number of shorter duration movies (less than 75 mins) post 2010. Overall, Short movies have been popular in last 10 years.

# * 5.7  What is the best time of the year when maximum content get added on the Netflix?

# In[337]:


month_year = df.groupby(['year_added' , 'month_added'])['show_id'].count().reset_index()


# In[338]:


plt.figure(figsize = (10,6))
sns.lineplot(data=month_year, x = 'year_added', y = 'show_id', hue='month_added')
plt.title('Year and Month of Adding Shows on Netflix')


# * The number of shows getting added is increasing with each year until 2020.
# * Also, months in the last quarter of the year (Oct-Dec) have more shows being added than the other months of the year. This could be because US has its festive season in Dec and India also has Diwali in Oct-Nov.

# In[ ]:





# * 5.8 Which countries are adding more number of content over the time?

# In[472]:


country_list = country_tb.country.value_counts().head(12).index
top_12_country = country_tb.loc[country_tb['country'].isin(country_list)]
country_year = top_12_country.merge(df , on = 'show_id')[['show_id','country_x' ,'type_x' , 'year_added' ]]
country_year.columns = ['show_id', 'country', 'type', 'year_added']


# In[ ]:


country_year = country_year.groupby(['country' , 'year_added'])['show_id'].count().reset_index()


# In[469]:


plt.figure(figsize = (10,6))
sns.lineplot(data = country_year , x = 'year_added' , y = 'show_id' , hue = 'country' , palette ='rainbow' )


# Observation : 
# United Stated have always added highset number of movies/TV shows over the time. Since 2016, India has seen spike in popularity of content and added more number of content, followed by United Kingdom at 3rd position.

# In[482]:


movie_type = country_year.loc[country_year.type == 'Movie'].groupby(['country' , 'year_added'])['show_id'].count().reset_index()
tv_type = country_year.loc[country_year.type == 'TV Show'].groupby(['country' , 'year_added'])['show_id'].count().reset_index()


# In[487]:


plt.figure(figsize = (10,6))
sns.lineplot(data = movie_type , x = 'year_added' , y = 'show_id' , hue = 'country' , palette ='rainbow' )


# In[485]:


plt.figure(figsize = (10,6))
sns.lineplot(data = tv_type , x = 'year_added' , y = 'show_id' , hue = 'country' , palette ='rainbow' )


# Observation: 
# It is observed that United States tops in both movies and TV Shows. India is at 2nd positon in movies but In TV shows United Kingdom is at 2nd position, followed by India ,South Korea , Australia. 
# It shows in countries like United Kingdom , South Korea , Australia TV Shows popularity is rising more than movies

# # Insights based on Non-Graphical and Visual Analysis 

# * Around 70% content on Netflix is Movies and around 30% content is TV shows.
# * The movies and TV shows uploading on the Netflix started from the year 2008, It had very lesser content till 2014. 
# * Year 2015 marks the drastic surge in the content getting uploaded on Netflix. It continues the uptrend since then and 2019 marks the highest number of movies and TV shows added on the Netflix. Year 2020 and 2021 has seen the drop in content added on Netflix, possibly because of Pandemic. But still , TV shows content have not dropped as drastic as movies. 
# * Since 2018, A drop in the movies is seen , but rise in TV shows is observed clearly.  Being in continuous uptrend , TV shows surpassed the movies count in mid 2020. It shows the rise in popularity of tv shows in recent years.
# * Netflix has movies from variety of directors. Around 4993 directors have their movies or tv shows on Netflix.
# * Netflix has movies from total 122 countries, United States being the highset contributor with almost 37% of all the content.
# * The release year for shows is concentrated in the range 2005-2021.
# * 50 mins - 150 mins is the range of movie durations, excluding potential outliers.
# * 1-3 seasons is the range for TV shows seasons, excluding potential outliers.
# * various ratings of content is avaialble on netfilx, for the various viewers categories like kids, adults , families. Highest number of movies and TV shows are rated TV-MA (for mature audiences).
# * Content in most of the ratings is available in lesser quanitity except in US. Ratings like TV-Y7 , TV-Y7 FV , PG ,TV-G , G , TV-Y , TV-PG are very less avaialble in all countries except US.
# * International Movies and TV Shows , Dramas , and Comedies are the top 3 genres on Netflix for both Movies and TV shows.
# * Mostly country specific popular genres are observed in each country. Only United States have a good mix of almost all genres. Eg. Korean TV shows (Korea), British TV Shows (UK), Anime features and Anime series (Japan) and so on.
# * Indian Actors have been acted in maximum movies on netflix. Top 5 actors are in India based on quantity of movies.
# *  Shorter duration movies have been popular in last 10 years.
# 
# 
# 

# # Business Insights 

# * Netflix have majority of content which is released after the year 2000. It is observed that the content older than year 2000 is very scarce on Netflix. Senior Citizen could be the target audience for such content, which is almost missing currently.
# * Maximum content (more than 80%) is 
#     * TV-MA - Content intended for mature audiences aged 17 and above.
#     * TV-14 - Content suitable for viewers aged 14 and above.
#     * TV-PG - Parental guidance suggested (similar ratings - PG-13 , PG)
#     * R - Restricted Content, that may not be suitable for viewers under age 17.
# 
# These ratings' movies target Matured and Adult audience. Rest 20 % of the content is for kids aged below 13.
# It shows that Netflix is currently serving mostly Mature audiences or Children with parental guidance.
# * Most popular genres on Netflix are International Movies and TV Shows , Dramas , Comedies, Action & Adventure, Children & Family Movies, Thrillers.
# * Maximum content of Netflix which is around 75% , is coming from the top 10 countries. Rest of the world only contributes 25% of the content. More countries can be focussed in future to grow the business.
# * Liking towards the shorter duration content is on the rise. (duration 75 to 150 minutes and seasons 1 to 3)
# This can be considered while production of new content on Netflix.
# * drop in content is seen across all the countries and type of content in year 2020 and 2021, possibly because of Pandemic.
# 
# 
# 

# # Recommendations

# * Very limited genres are focussed in most of the countries except US. It seems the current available genres suits best for US and few countries but maximum countries need some more genres which are highly popular in the region.
# eg. Indian Mythological content is highly popular. We can create such more country specific genres and It might also be liked acorss the world just like Japanese Anime.
# 

# * Country specific insights - The content need to be targetting the demographic of any country. Netflix can produce higher number of content in the perticular rating as per demographic of the country. Eg.
#     * The country like India , which is highly populous , has maximum content available only in three rating TV-MA, TV-14 , TV-PG. It is unlikely to serve below 14 age and above 35 year age group . 

# ![image.png](attachment:image.png)

# * Country Japan have only 3 rating of content largely served - TV-MA, TV-14 , TV-PG.
# Japan have high population of age above 60, and this can be served by increasing the content suitable for this age group.

# ![image.png](attachment:image.png)

# *  Netflix is currently serving mostly Mature audiences or Children with parental guidance. It have scope to cater other audiences as well such as familymen , Senior citizen , kids of various age etc.
