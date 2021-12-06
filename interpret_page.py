#imports 
import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from predict_page import loads_model
#training dataset 
from sklearn.model_selection import train_test_split

#convert dict of genres to list 
import re
import json
def dict_lists(dataFrame,src_column):
    new_vals = []
    for value in dataFrame[src_column]:
        if value == value:
            # getting list of substrings
            jsons_list = re.findall('{.*?}', value)
            # getting dict_list for this value
            dicts_list = []
            for json_val in jsons_list:
                dicts_list.append(json.loads(json_val))
            new_vals.append(dicts_list)
        else:
            new_vals.append(np.nan)
    return new_vals

#make the list of genres into seperate columns 
def list_category(category):
    for i in range(0, 4803):
        for j in range(0, 7):
            if category.iloc[i][j] != None:
                category.iloc[i][j] = category.iloc[i][j].get('name')   

#change the numbers into name of the month 
def clean_month(x):
    if "12" in x:
        return "December"
    if "11" in x:
        return "November"
    if "10" in x:
        return "October"
    if "1" in x:
        return "January"
    if "2" in x:
        return "Feburary"
    if "3" in x:
        return "March"
    if "4" in x:
        return "April"
    if "5" in x:
        return "May"
    if "6" in x:
        return "June"
    if "7" in x:
        return "July"
    if "8" in x:
        return "August"
    if "9" in x:
        return "September"


#remove the languages where there are only a few values 
def shorten_categories(categories, cutoff):
  categorical_map = {}
  for i in range(len(categories)):
    if categories.values[i] >= cutoff:
      categorical_map[categories.index[i]] = categories.index[i]
    else: 
      categorical_map[categories.index[i]] = "Other"
  return categorical_map


def clean_language(x):
    if "en" in x:
        return "English"
    if "fr" in x: 
        return "French"
    if "es" in x: 
        return "Spanish"
    if "zh" in x:
        return "Chinese"
    if "de" in x: 
        return "German"
    if "ja" in x: 
        return "Japanese"
    if "hi" in x: 
        return "Hindi"
    if "cn" in x:
        return "Chinese"
    if "ru" in x: 
        return "Russian"
    if "ko" in x: 
        return "Korean"
    if "it" in x: 
        return "Italian"

#remove the languages where there are only a few values 
def shorten_categories(categories, cutoff):
  categorical_map = {}
  for i in range(len(categories)):
    if categories.values[i] >= cutoff:
      categorical_map[categories.index[i]] = categories.index[i]
    else: 
      categorical_map[categories.index[i]] = "Other"
  return categorical_map

@st.cache

def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv")

    df_genres = pd.DataFrame(dict_lists(df,'genres'))
    df_production_company = pd.DataFrame(dict_lists(df,'production_companies'))
    df_production_country = pd.DataFrame(dict_lists(df,'production_countries'))

    list_category(df_genres)
    list_category(df_production_company)
    list_category(df_production_country)

    #add genre column to original dataframe
    df['top_genre'] = df_genres[0]
    df["top_production_company"] = df_production_company[0]
    df["top_production_country"] = df_production_country[0]

    #only want to keep a few columns 
    df_sample = df[["budget", "top_genre","revenue", "top_production_company", 
    "top_production_country", "original_language", "popularity", "runtime"]]

    df_sample['release_day'] = df['release_date'].str[8:]
    df_sample['release_month'] = df['release_date'].str[5:7]
    df_sample['release_year'] = df['release_date'].str[0:4]

    df_sample = df_sample.dropna()
    df_sample.isnull().sum()

    df_sample = df_sample[df_sample["top_genre"] != "TV Movie"]
    df_sample = df_sample[df_sample["top_genre"] != "Foreign"]
    df_sample = df_sample[df_sample["budget"] != 0]
    df_sample = df_sample[df_sample["budget"] >= 5000]

    df_sample['release_month'] = df_sample['release_month'].apply(clean_month)

    language_map = shorten_categories(df_sample['original_language'].value_counts(), 10)
    df_sample["original_language"] = df_sample['original_language'].map(language_map)
    df_sample['original_language'].value_counts()

    production_company_map = shorten_categories(df_sample['top_production_company'].value_counts(), 10)
    df_sample['top_production_company'] = df_sample['top_production_company'].map(production_company_map)
    df_sample['top_production_company'].value_counts()

    df_sample['genre'] = df_sample['top_genre']

    #some of the data is a number, but some is a string. 
    #So, I need to transform the string values into numbers 
    #that the model can understand
    from sklearn.preprocessing import LabelEncoder
    le_top_genre = LabelEncoder()
    df_sample['top_genre'] = le_top_genre.fit_transform(df_sample['top_genre'])
    le_top_production_company = LabelEncoder()
    df_sample['top_production_company'] = le_top_production_company.fit_transform(df_sample['top_production_company'])
    le_top_production_country = LabelEncoder()
    df_sample['top_production_country'] = le_top_production_country.fit_transform(df_sample['top_production_country'])                                                                             
    le_original_language = LabelEncoder()
    df_sample['original_language'] = le_original_language.fit_transform(df_sample['original_language'])                                                                             
    le_release_month = LabelEncoder()
    df_sample['release_month'] = le_release_month.fit_transform(df_sample['release_month'])  

    df_sample['profit'] = df_sample["revenue"] - df_sample["budget"]
    return df_sample

#set the loaded data 
df = load_data()

#display the interpretation page 
def show_interpret_page():

    #X = features and y = what we want to predict
    X = df.drop(['revenue', 'release_year', 'original_language', 'top_production_country', 'genre'], axis=1) #noram
    y = df['revenue']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    #DecisionTreeRegressor 
    #combines multiple decision trees into a forest
    from sklearn.ensemble import RandomForestRegressor
    random_forest_reg = RandomForestRegressor(random_state=0)
    random_forest_reg.fit(X_train, y_train)

    y_pred = random_forest_reg.predict(X_test)

    #the error is a bit better here 
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import numpy as np
    error = np.sqrt(mean_squared_error(y_test, y_pred))
    #print("${:,.02f}".format(error))

    #calculate accuracy 
    acc = random_forest_reg.score(X_test,y_test) * 100

    st.title("Interpret Prediction Model")
    st.sidebar.write("This application utilizes a Random Tree Regression Model.")

    metrics = st.container()

    col1, col2 = metrics.columns(2)

    col1.metric(label="Accuracy", value = acc)
    col2.metric(label="Potenial Error", value = error)

    graph_container = st.container()

    txt = st.sidebar.text_area('Text to analyze', height=400)

    #download button to download interpretation notes 
    st.sidebar.download_button(label = 'Download interpretation notes', data =txt, file_name='boxOffice_notes.txt')

    graph = graph_container.selectbox("Choose your visualization:", ("Genre/Budget", "Genre/Profit", 
    "Genre/Popularity", "Release Month/Revenue", "Release Day/Revenue", "Runtime/Revenue", "DataFrame"))

    if graph == "Genre/Budget":
        graph_container.write(""" #### Avg. Popularity on Genre """)
        data = df.groupby(["genre"])["budget"].mean().sort_values(ascending=True)
        graph_container.bar_chart(data)
        graph_container.write('When make decisions on your next movie, it is important to consider the current popularity. Based on the visualization below, the most popular genres are Animation and Adventure.')
    if graph == "Genre/Profit":
        graph_container.write(""" #### Avg. Profit Based on Genre """)
        data1 = df.groupby(["genre"])["profit"].mean().sort_values(ascending=True)
        graph_container.area_chart(data1)
    if graph == "Genre/Popularity":
        graph_container.write(""" #### Avg. Popularity on Genre """)
        data2 = df.groupby(["genre"])["popularity"].mean().sort_values(ascending=True)
        graph_container.bar_chart(data2)
    if graph == "Release Month/Revenue":
        graph_container.write(""" #### Avg. Popularity on Genre """)
        data3 = df.groupby(["release_month"])["revenue"].mean().sort_values(ascending=True)
        graph_container.bar_chart(data3)
    if graph == "Release Day/Revenue":
        graph_container.write(""" #### Avg. Popularity on Genre """)
        data4 = df.groupby(["release_day"])["revenue"].mean().sort_values(ascending=True)
        graph_container.area_chart(data4)
    if graph == "Runtime/Revenue":
        graph_container.write(""" #### Avg. Popularity on Genre """)
        data5 = df.groupby(["runtime"])["revenue"].mean().sort_values(ascending=True)
        graph_container.area_chart(data5)
    if graph == "DataFrame":
        #dataframe
        st.dataframe(data = df)
        
    #could maybe change the model that they use with checkbox