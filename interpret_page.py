import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

#convert dict to list 
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

def list_category(category):
    for i in range(0, 4803):
        for j in range(0, 7):
            if category.iloc[i][j] != None:
                category.iloc[i][j] = category.iloc[i][j].get('name')   


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

    #X = features and y = what we want to predict
    df_sample= df_sample.drop(['original_language', 'top_production_country', 'top_production_company'], axis=1) #noram

    df_sample['profit'] = df_sample["revenue"] - df_sample["budget"]
    return df_sample

df = load_data()

def show_interpret_page():
    metric_container = st.container()
    col1, col2, col3 = metric_container.columns(3)
    col1.metric("Accuracy", "96.3%", "12")
    col2.metric("Percision", calc_percision())
    col3.metric("Error Rate", "$4,218", "7")
    st.title("Interpret Prediction Model")
    st.write("This model ")

    graph_container = st.container()
    #heatmap 
    #bar graph 
    #graph_container.write(""" #### Avg. Profit Based on Genre """)
    #data = df.groupby(["top_genre"])["profit"].mean().sort_values(ascending=True)
    #graph_container.area_chart(data)

    txt = st.sidebar.text_area('Text to analyze', height=400)

    st.sidebar.download_button('Download interpretation notes', txt)
    


    graph_container.write(""" #### Avg. Popularity on Genre """)
    data = df.groupby(["top_genre"])["budget"].mean().sort_values(ascending=True)
    data1 = df.groupby(["top_genre"])["revenue"].mean().sort_values(ascending=True)
    data2 = df.groupby(["top_genre"])["popularity"].mean().sort_values(ascending=True)
    data3 = df.groupby(["release_month"])["revenue"].mean().sort_values(ascending=True)
    data4 = df.groupby(["release_day"])["revenue"].mean().sort_values(ascending=True)
    data5 = df.groupby(["runtime"])["revenue"].mean().sort_values(ascending=True)
    graph_container.bar_chart(data)
    graph_container.bar_chart(data1)
    graph_container.bar_chart(data2)
    graph_container.bar_chart(data3)
    graph_container.area_chart(data4)
    graph_container.area_chart(data5)

    #dataframe
    st.dataframe(data = df)
    #could maybe change the model that they use