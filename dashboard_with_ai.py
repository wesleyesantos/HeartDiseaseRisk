import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import pickle
import json
import sklearn
import plotly.graph_objs as go
import random
# import streamviz as sv
import matplotlib.pyplot as plt
from joblib import dump, load
import dice_ml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from streamlit_option_menu import option_menu
from helperfunctions import *

def load_dashboard_with_ai():
    DATASET_PATH = "heart_2020_cleaned.parquet"
    LOG_MODEL_PATH = "logistic_regression.pkl"
    #Gets random row from the dataset
    total_rows= 319796
    if "random_row_index" not in st.session_state:
        # If not, generate a new random row index and store it in the session state
        st.session_state.random_row_index = random.randint(0, total_rows - 1)


    random_person_1 = ["128868","Obese (30.0 <= BMI < +Inf)","Yes","No","No",12.0,10.0,"Yes","Male","50-54","White","Yes","No","Poor",6.0,"No","No","No"]
    random_person_3 = ["222460","Overweight (25.0 <= BMI < 30.0)","Yes","No","No",10.0,30.0,"Yes","Male","75-79","White","Yes","No","Fair",10.0,"No","No","No"]
    random_person_5 = ["44","Overweight (25.0 <= BMI < 30.0)","Yes","Yes","Yes",10.0,30.0,"Yes","Female","80 or older","White","Yes","No","Fair",7.0,"No","No","No"]
    random_person_list = [random_person_1, random_person_3, random_person_5]


    if "chosen_person" not in st.session_state:
        random_number = random.randint(0, 2)

        chosen_person = random_person_list[random_number]
        st.session_state.chosen_person = chosen_person
    chosen_person = st.session_state.chosen_person
    patient_num = chosen_person[0]
    BMI = chosen_person[1]
    smokingcat = chosen_person[2]
    alcohol = chosen_person[3]
    strokecat = chosen_person[4]
    physicalhealth = chosen_person[5]
    diffwalk = chosen_person[7]
    gender = chosen_person[8]
    age = chosen_person[9]
    diabeticcat = chosen_person[11]
    physicalactivity = chosen_person[12]
    genhealth = chosen_person[13]
    sleeptime = chosen_person[14]
    asthma = chosen_person[15]




    def chart_spec_pie(category_counts_dicts):
        return {
        "data": {
            "values": category_counts_dicts  # Use static list of dicts
        },
        "width": 200,  # Set the width of the chart
        "height": 200, # Set the height of the chart
        "mark": "arc",
        "encoding": {
            "theta": {"field": "Count", "type": "quantitative"},
            "color": {"field": "Category", "type": "nominal", "legend": {"title": "Categories"}}
        },
        }
        
    col1, col2 = st.columns([3,3])
    # contcol1 = col1.container()
    contcontcol1 = col1.container(border=True)
    contcontcol1.subheader(f"Hello, Patient {patient_num}")
    contcontcol1.markdown("""
                    Welcome to your health dashboard. 
                    Here you can find all the information about your health. Disclaimer, the information you can find here is based on a sample of 319,796 people thus the data might not represent the entire population""")
    contcol2 = col2.container(border=True)







    contcol2.markdown("<p style='text-align: center;' > Your calculated risk is</p>", unsafe_allow_html=True)
    if(st.session_state.prediction_bool == 0):
        contcol2.markdown("<h1 style='text-align:center;font-size:3rem; padding:0rem; color:green;'>" + st.session_state.prediction + "%</h1>", unsafe_allow_html=True)
        contcol2.markdown("<p style='text-align: center;' >Considered Healthy</p>", unsafe_allow_html=True)
    else:
        contcol2.markdown("<h1 style='text-align:center;font-size:3rem; padding:0rem; color:red;'>" + st.session_state.prediction + "%</h1>", unsafe_allow_html=True)
        contcol2.markdown("<p style='text-align: center;' >Considered Unhealthy</p>", unsafe_allow_html=True)


    st.markdown(
        """
        <style>
            [data-testid=stSidebar] [data-testid=stImage]{
                text-align: center;
                display: block;
                margin-left: auto;
                margin-right: auto;
                width: 100%;
            }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] {
                width: 22rem !important; # Set the width to your desired value
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Using object notation
    st.sidebar.image('pictures/stock_placeholder.jpg', width=100)
    st.sidebar.markdown("<h1 style='text-align: center;' >Patient " + patient_num + "</h1>", unsafe_allow_html=True)
    sidecont1, sidecont2 = st.sidebar.columns([3, 3])

    custom_css = """
    <style>
    /* Targeting all Markdown widgets inside the sidebar to reduce their margin */
    [data-testid="stSidebar"] .stMarkdown {
    margin-bottom: -15px;
    margin-top: -15px;
    }
    /* Adjusting padding inside the containers - might not work as expected since specific targeting like this is limited */
    [data-testid="stSidebar"] .stBlock {
    padding: 50px;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    with st.sidebar:
        with st.container(border=True):
            st.markdown("<h2 style='text-align: center;' >Age </h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;' >" + age +"<p>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<h2 style='text-align: center;' >BMI </h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;' >" + BMI +"<p>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<h2 style='text-align: center;' >Gender </h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;' >" + gender +"<p>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<h2 style='text-align: center;' >General Health </h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;' >" + genhealth +"<p>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<h2 style='text-align: center;' >Asthma </h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;' >" + asthma +"<p>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<h2 style='text-align: center;' >Difficulty Walking </h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;' >" + diffwalk +"<p>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<h2 style='text-align: center;' >Diabetic </h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;' >" + diabeticcat +"<p>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<h2 style='text-align: center;' >Sleep Time </h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;' >" + str(sleeptime) +"<p>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<h2 style='text-align: center;' >Smoking </h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;' >" + smokingcat +"<p>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<h2 style='text-align: center;' >Stroke </h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;' >" + strokecat +"<p>", unsafe_allow_html=True)
        



    with col1.container():
        st.subheader("Your results")



    with col1.container():
        option = st.selectbox(
            "What Graph would you like to see?",
            ("BMI", "Smoking", "Sleep Time", "General Health", "Alcohol Drinking", "Stroke", "Difficulty Walking", "Diabetic"),
            placeholder="Select a Graph to display",
        )

    # Assuming 'heart' is your DataFrame and 'col1' is a defined Streamlit container
    if option == "BMI":    
        with col1.container(border=True):
            bar_chart_spec = {
        "layer": [
            {
                "mark": "bar",
                "encoding": {
                    "x": {
                        "field": "BMICategory",
                        "type": "nominal",
                        "title": "BMI Category",
                        "axis": {"labelAngle": 0},
                        "sort": ["Underweight (BMI < 18.5)", "Normal weight (18.5 <= BMI < 25.0)", "Overweight (25.0 <= BMI < 30.0)", "Obese (30.0 <= BMI < +Inf)"]
                    },
                    "y": {
                        "field": "count",
                        "type": "quantitative",
                        "title": "Number of People"
                    }
                }
            },
            {
        "mark": {"type": "rule", "color": "red", "size": 2},  # Red line configuration
        "encoding": {
            "x": {
                "field": "BMICategory",  # Ensuring this matches the bar chart's field
                "type": "nominal",
                "datum": BMI  # The specific category you're highlighting
            },
            "tooltip": {
                "value": f"Your BMI category is {BMI}"  # Custom tooltip message
            }
        }
    }
        ]
        }

        # Manually set values for each category
            bmi_values = {
            "Underweight (BMI < 18.5)": 5110,
            "Normal weight (18.5 <= BMI < 25.0)": 97331,
            "Overweight (25.0 <= BMI < 30.0)": 114512,
            "Obese (30.0 <= BMI < +Inf)": 102842
            }
    # Create a DataFrame from the static values
            static_data = pd.DataFrame(list(bmi_values.items()), columns=['BMICategory', 'count'])
            st.markdown(f"The red line is your BMI category: :red[{BMI}]]")
            st.vega_lite_chart(static_data, bar_chart_spec, use_container_width=True)

    if option == "General Health":
        with col1.container(border=True):
            bar_chart_spec = {
        "layer": [
            {
                "mark": "bar",
                "encoding": {
                    "x": {
                        "field": "GeneralHealth",
                        "type": "nominal",
                        "title": "General Health",
                        "axis": {"labelAngle": 0},
                        "sort": ["Poor", "Fair", "Good", "Very good", "Excellent"]
                    },
                    "y": {
                        "field": "count",
                        "type": "quantitative",
                        "title": "Number of People"
                    }
                }
            },
            {
        "mark": {"type": "rule", "color": "red", "size": 2},  # Red line configuration
        "encoding": {
            "x": {
                "field": "GeneralHealth",  # Ensuring this matches the bar chart's field
                "type": "nominal",
                "datum": genhealth  # The specific category you're highlighting
            },
            "tooltip": {
                "value": f"Your general health is {genhealth}"  # Custom tooltip message
            }
        }
    }


        ]
        }
            gen_values = {
            "Excellent": 66842,
            "Very good": 113858,
            "Good": 93128,
            "Fair": 34677,
            "Poor": 11289
            }   
    # Create a DataFrame from the static values
            static_data = pd.DataFrame(list(gen_values.items()), columns=['GeneralHealth', 'count'])
            st.markdown(f"The red line is your General Health category: :red[{genhealth}]")
            st.vega_lite_chart(static_data, bar_chart_spec, use_container_width=True)





    if option == "Diabetes":
        category_counts_dicts = [
        {"Category": "No", "Count": 269653},
        {"Category": "Yes", "Count": 40802},
        {"Category": "No, borderline diabetes", "Count": 6781},
        {"Category": "Yes (during pregnancy)", "Count": 2559}
        ]

        pie_chart_spec = chart_spec_pie(category_counts_dicts)
        # Static values for the calculation
        no_smoking = category_counts_dicts[0]['Count']
        yes_smoking = category_counts_dicts[1]['Count']
        percentage = round((yes_smoking / (yes_smoking + no_smoking)) * 100, 2)
        # Adjusted for example completeness
        with col1.container(border=True):
            st.subheader("Diabetes")        
            # if diabeticcat == "Yes":
            #     st.markdown(f"<p style='color: red;'>You are part of the {percentage}% that has had a stroke.</p>", unsafe_allow_html=True)
            # if diabeticcat == "No":
            #     st.markdown(f"<p style='color: green;'>You are part of the {100 - percentage}% that hasn't had a stroke.</p>", unsafe_allow_html=True)
            # if diabeticcat == "No, borderline diabetes":
            #     st.markdown(f"<p style='color: green;'>You are part of the {100 - percentage}% that hasn't had a stroke.</p>", unsafe_allow_html=True)
            # if diabeticcat == "Yes (during pregnancy)":
            #     st.markdown(f"<p style='color: green;'>You are part of the {100 - percentage}% that hasn't had a stroke.</p>", unsafe_allow_html=True)
            st.vega_lite_chart(pie_chart_spec, use_container_width=True)
        
    if option == "Stroke":
        category_counts_dicts = [
        {"Category": "No", "Count": 307726},
        {"Category": "Yes", "Count": 12069}
        ]

        pie_chart_spec = chart_spec_pie(category_counts_dicts)
        # Static values for the calculation
        no_smoking = category_counts_dicts[0]['Count']
        yes_smoking = category_counts_dicts[1]['Count']
        percentage = round((yes_smoking / (yes_smoking + no_smoking)) * 100, 2)
        # Adjusted for example completeness
        with col1.container(border=True):
            st.subheader("Stroke")        
            if strokecat == "Yes":
                st.markdown(f"<p style='color: red;'>You are part of the {percentage}% that has had a stroke.</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='color: green;'>You are part of the {100 - percentage}% that hasn't had a stroke.</p>", unsafe_allow_html=True)
            st.vega_lite_chart(pie_chart_spec, use_container_width=True)
        

    if option == "Alcohol Drinking":
        category_counts_dicts = [
        {"Category": "No", "Count": 298018},
        {"Category": "Yes", "Count": 21777}
        ]

        pie_chart_spec = chart_spec_pie(category_counts_dicts)
        # Static values for the calculation
        no_smoking = category_counts_dicts[0]['Count']
        yes_smoking = category_counts_dicts[1]['Count']
        percentage = round((yes_smoking / (yes_smoking + no_smoking)) * 100, 2)
        # Adjusted for example completeness
        with col1.container(border=True):
            st.subheader("Alcohol Drinking")        
            if alcohol == "Yes":
                st.markdown(f"<p style='color: red;'>You are part of the {percentage}% that drinks.</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='color: green;'>You are part of the {100 - percentage}% that doesn't drink.</p>", unsafe_allow_html=True)
            st.vega_lite_chart(pie_chart_spec, use_container_width=True)

    if option == "Difficulty Walking":
        category_counts_dicts = [
        {"Category": "No", "Count": 275385},
        {"Category": "Yes", "Count": 44410}
        ]

        pie_chart_spec = chart_spec_pie(category_counts_dicts)
        # Static values for the calculation
        no_diff = category_counts_dicts[0]['Count']
        yes_diff = category_counts_dicts[1]['Count']
        percentage = round((yes_diff / (yes_diff + no_diff)) * 100, 2)
        # Adjusted for example completeness
        with col1.container(border=True):
            st.subheader("Difficulty Walking")        
            if diffwalk == "Yes":
                st.markdown(f"<p style='color: red;'>You are part of the {percentage}% that has difficulty walking.</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='color: green;'>You are part of the {100 - percentage}% that doesn't have difficulty walking.</p>", unsafe_allow_html=True)
            st.vega_lite_chart(pie_chart_spec, use_container_width=True)


    if option == "Smoking":
        category_counts_dicts = [
        {"Category": "No", "Count": 187887},
        {"Category": "Yes", "Count": 131908}
        ]

        pie_chart_spec = chart_spec_pie(category_counts_dicts)
        # Static values for the calculation
        no_smoking = category_counts_dicts[0]['Count']
        yes_smoking = category_counts_dicts[1]['Count']
        percentage = round((yes_smoking / (yes_smoking + no_smoking)) * 100, 2)
        # Adjusted for example completeness
        with col1.container(border=True):
            st.subheader("Smoking")
            st.markdown("<p>Smoking triples your heart disease risk.</p>", unsafe_allow_html=True)
            
            if smokingcat == "Yes":
                st.markdown(f"<p style='color: red;'>You are part of the {percentage}% that smokes</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='color: green;'>You are part of the {100 - percentage}% that doesn't smoke</p>", unsafe_allow_html=True)
            st.vega_lite_chart(pie_chart_spec, use_container_width=True)


    with col2.container(border=True):
        st.markdown("<p>Here you can ask any questions you have about your health. The AI will try to answer them to the best of its ability.</p>", unsafe_allow_html=True)
        st.markdown("""<iframe src="https://vanherwegentim-chatbot-app-ci68bm.streamlit.app/?embed_options=disable_scrolling,show_padding,show_colored_line,show_toolbar,show_footer&embed=true" height="550" style="width: 100%; border: none;"></iframe>""", unsafe_allow_html=True)


    