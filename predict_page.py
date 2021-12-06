#imports 
import streamlit as st
import pickle 
import numpy as np 
import pandas as pd

#load model from Jupyter Notebook 
def loads_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data
#set model 
data = loads_model()

#set Decision Tree Regression model and label encoders 
regressor = data["model"]
le_top_genre = data["le_top_genre"]
le_top_production_company = data["le_top_production_company"]
le_top_production_country = data["le_top_production_country"]
le_original_language = data["le_original_language"]
le_release_month = data["le_release_month"]

#show the prediction page 
def show_predict_page(): 
    #create different widgets 
    st.title("Box Office Gross Revenue Prediction")
    #st.write("""### We need some information to predict the revenue""")

    #add select options for the features 

    #genres categories 
    genres = {
        "Drama",
        "Comedy",
        "Action", 
        "Adventure", 
        "Horror", 
        "Crime",
        "Thriller", 
        "Animation", 
        "Fantasy", 
        "Romance", 
        "Science Fiction", 
        "Documentary", 
        "Family", 
        "Mystery", 
        "Music", 
        "Western", 
        "History", 
        "War"
    }

    #production companies categories - too many options so NOT included at this time
    production_companies = {
        "Paramount Pictures", 
        "Universal Pictures"
        "Columbia Pictures", 
        "Twentieth Century Fox Film Corporation",
        "New Line Cinema",
        "Walt Disney Pictures",
        "Miramax Films",
        "United Artists",
        "Village Roadshow Pictures",
        "Warner Bros.",
        "Columbia Pictures Corporation",
        "Fox Searchlight Pictures",
        "DreamWorks SKG ",
        "Metro-Goldwyn-Mayer (MGM)",
        "Summit Entertainment",
        "TriStar Pictures",
        "Touchstone Pictures",
        "Regency Enterprises",
        "The Weinstein Company",
        "Lionsgate",
        "The Weinstein Company",
        "Lionsgate",
        "Imagine Entertainment",
        "Lions Gate Films",
        "Dimension Films",
        "Lakeshore Entertainment",
        "BBC Films",
        "Fox 2000 Pictures",
        "Castle Rock Entertainment",
        "Ingenious Film Partners",
        "Hollywood Pictures",
        "Spyglass Entertainment",
        "Screen Gems",
        "Sony Pictures Classics",
        "Legendary Pictures",
        "StudioCanal",
        "Dune Entertainment",
        "Alcon Entertainment",
        "Lucasfilm ",
        "Studio Babelsberg",
        "Orion Pictures",
        "DreamWorks Animation",
        "Amblin Entertainment",
        "France 2 Cinéma",
        "ine Line Features",
        "Destination Films ",
        "Double Feature Films",
        "Revolution Studios",
        "Eon Productions",
        "Jerry Bruckheimer Films",
        "DC Comics",
        "WingNut Films",
        "Marvel Studios",
        "American Zoetrope",
        "Relativity Media",
        "Impact Pictures",
        "Original Film",
        "Silver Pictures",
        "Other",
    }

    #production countries categories - not very releavant to movie characterization so NOT included at this time
    production_countries = {
        "United States of America", 
        "United Kingdom",
        "Canada",
        "Germany",
        "France",
        "Australia",
        "China",
        "India",
        "Japan",
        "Spain",
        "Italy",
        "Hong Kong",
        "Ireland",
        "New Zealand",
        "Mexico",
        "Czech Republic",
        "Belgium",
        "Denmark",
        "South Korea",
        "Brazil",
        "Russia",
        "Switzerland",
        "Other",
    }

    #original Langugae categories - not relevant to movie characterization so NOT included at this time
    languages = {
        "English", 
        "French", 
        "Spanish", 
        "Chinese", 
        "German", 
        "Japanese", 
        "Hindi", 
        "Russian",
        "Korean", 
        "Italian", 
        "Other"
    }

    #release months 
    months = {
        "January", 
        "Feburary", 
        "March", 
        "April", 
        "May", 
        "June", 
        "July", 
        "August", 
        "September", 
        "October", 
        "November", 
        "December"
    }
    
    #set container for webpage 
    metric_container = st.container()
    metric_container.write("Enter information to get insights into your film's future:")

    #checkbox to filter for only movies release during 2020 and 2021
    pandemic_movies = metric_container.checkbox("Include only movies released during the pandemic in prediction model")

    #show the budget (inputed by the user) and the predicted revenue 
    col1, col2 = metric_container.columns(2)
    test = pd.DataFrame({
        'Profit': [0, 50000000], 
        'Budget': [20000000, 0]
    })

    #show the budget and predicted revenue bar chart 
    graph = st.bar_chart(test)
    
    #sidebar features 
    budget = st.sidebar.text_input('Budget', 'What is the budget (no commas)?')
    genre = st.sidebar.multiselect("Genre (choose only one)", genres)
    release_month = st.sidebar.selectbox("Release Month", months)
    release_day = st.sidebar.number_input("Release Day", min_value=1, max_value=31, value=5, step=1)
    popularity = st.sidebar.slider("Popularity", 0.0, 10.0, 3.0)
    runtime = st.sidebar.slider("Runtime", 0, 200, 50)
    #not included at this time
    production_company= st.multiselect("Production Company (choose only one):", production_companies)
    #production_country = st.selectbox("Country of Production", production_countries)
    #original_language = st.selectbox("Film Original Language", languages)
    
    #button for the prediction 
    ok = st.button("Predict Revenue")

    #trigger to make the prediction when button is pressed 
    if ok: 
        #start the prediction 
        X = np.array([[budget ,genre[0], production_company[0], popularity, runtime, release_day, release_month]])
        X[:, 1] = le_top_genre.transform(X[:, 1])
        X[:, 2] = le_top_production_company.transform(X[:,2])
        #X[:, 3] = le_top_production_country.transform(X[:, 3])
        #X[:, 4] = le_original_language.transform(X[:,4])
        X[:, 6] = le_release_month.transform(X[:, 6])

        #make prediction 
        y_pred = regressor.predict(X)

        #calculate overall profit of the film 
        profit = y_pred[0] - int(budget)

        #change format of budget, revenue, and profit to include commas 
        bud = "{:,}".format(int(budget))
        rev = "{:,}".format(int(f"{y_pred[0]:.0f}"))
        prof = "{:,}".format(int(f"{profit:.0f}"))

        #display budget and revenue metrics 
        col1.metric(label="Budget", value="$" + str(bud))
        col2.metric(label="Predicted Gross Revenue", value="$" + str(rev), delta=str(prof))

        #show bar chart for budget and predict revenue 
        test = pd.DataFrame({
        'Revenue': [0, y_pred[0]], 
        'Budget': [int(budget), 0]
        })
        graph.bar_chart(test)
