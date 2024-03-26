import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import pickle
import json
import sklearn
import random
# import streamviz as sv
from joblib import dump, load
import joblib
import dice_ml
from streamlit_option_menu import option_menu
from helperfunctions import *
from langchain.tools import BaseTool, StructuredTool, tool
from sklearn.base import BaseEstimator, TransformerMixin


# from joblib import load
import pandas as pd
from langchain.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ChatMessageHistory

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage



st.set_page_config(layout="wide")

selected = option_menu(
    menu_title=None,
    options=["Home", "Dashboard", "About"],
    icons=["house", "book", "envelope"],
    menu_icon="cast",
    orientation="horizontal",
    default_index=1,
    styles={
        "container":{"max-width":"100%", "padding":"0"},
        
    }
)
st.markdown(
        """
            <style>
                .appview-container .main .block-container {{
                    padding-top: {padding_top}rem;
                    padding-bottom: {padding_bottom}rem;
                    }}

            </style>""".format(
            padding_top=3, padding_bottom=3
        ),
        unsafe_allow_html=True,
    )


# from streamlit_float import *

# float_init(theme=True, include_unstable_primary=False)

if selected == "Home":
    col1, col2, _ = st.columns([1, 2,1])
    col2.title("Heart Health Advisor")
    col2.subheader("Welcome to the Heart Risk Assessment Tool")
    col2.markdown("""This tool aims to assist you in comprehending your risk of heart disease. Currently, rather than utilizing your personal information to evaluate heart disease risk, the tool employs data from a preset patient profile. This approach is adopted because handling confidential information necessitates extensive approvals from regulatory bodies. To interact with the Chatbot, please navigate to the dashboard. While you can inquire about various topics, for optimal results, we recommend focusing your questions on heart disease risk.

""")
    with col1.container():
        st.image("images/handholding.png")

if selected == "Dashboard":

    
    DATASET_PATH = "heart_2020_cleaned.parquet"
    LOG_MODEL_PATH = "logistic_regression.pkl"
    log_model = pickle.load(open(LOG_MODEL_PATH, "rb"))

    
    #Gets random row from the dataset
    total_rows= 319796
    st.sidebar.image('pictures/stock_placeholder.jpg', width=100)
    st.sidebar.markdown("<h1 style='text-align: center;' >Patient </h1>", unsafe_allow_html=True)
    



    option = st.sidebar.selectbox(
    'Patient',
    ('44', '222460','128868'), index=2, label_visibility="collapsed")
    if(option == '44'):
        num = 2
    if(option == '222460'):
        num = 1
    if(option == '128868'):
        num = 0
    st.session_state.num = num
    
    if 'previous_num' not in st.session_state:
        st.session_state.previous_num = num
    else:
        if st.session_state.num != st.session_state.previous_num:
            st.session_state.messages = [{"content": "Hello! How can I assist you today? I can answer all your questions about your heart disease risk. Als je mij aanspreekt in het Nederlands, kan ik je ook in het Nederlands antwoorden.", "role": "assistant"}]
            st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
            memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
            memory.load_memory_variables({})
            st.session_state.previous_num = st.session_state.num


    random_person_1 = ["128868","Obese (30.0 <= BMI < +Inf)","Yes","No","No",12.0,10.0,"Yes","Male","50-54","White","Yes","No","Poor",6.0,"No","No","No"]
    random_person_3 = ["222460","Overweight (25.0 <= BMI < 30.0)","Yes","Yes","No",10.0,30.0,"Yes","Male","75-79","White","Yes","No","Poor",10.0,"No","No","No"]
    random_person_5 = ["44","Overweight (25.0 <= BMI < 30.0)","No","Yes","No",10.0,30.0,"Yes","Female","65-69","White","Yes","No","Poor",4.0,"Yes","Yes","Yes"]
    random_person_list = [random_person_1, random_person_3, random_person_5]


    random_number = num

    chosen_person = random_person_list[random_number]
    st.session_state.chosen_person = chosen_person
    patient_num = chosen_person[0]
    BMI = chosen_person[1]
    smokingcat = chosen_person[2]
    alcohol = chosen_person[3]
    strokecat = chosen_person[4]
    physicalhealth = chosen_person[5]
    mentalhealth = chosen_person[6]
    diffwalk = chosen_person[7]
    gender = chosen_person[8]
    age = chosen_person[9]
    race = chosen_person[10]
    diabeticcat = chosen_person[11]
    physicalactivity = chosen_person[12]
    genhealth = chosen_person[13]
    sleeptime = chosen_person[14]
    asthma = chosen_person[15]
    kidneydisease = chosen_person[16]
    skincancer = chosen_person[17]

    @st.cache_data(persist=True)
    def load_dataset() -> pd.DataFrame:
        # Assuming you have converted your dataset to Parquet format
        # and updated the DATASET_PATH to point to the .parquet file
        parquet_file_path = DATASET_PATH  # Update this to your Parquet file path
        heart_df = pd.read_parquet(parquet_file_path)
        return heart_df



    def chart_spec_pie(category_counts_dicts):
        return {
        "data": {
            "values": category_counts_dicts  # Use static list of dicts
        },
        "width": 200,  # Set the width of the chart
        "height": 150, # Set the height of the chart
        "mark": "arc",
        "encoding": {
            "theta": {"field": "Count", "type": "quantitative"},
            "color": {"field": "Category", "type": "nominal", "legend": {"title": "Categories"}}
        },
        }
        
    col1, col2 = st.columns([2,3])
    with col1.container(border=True):
        st.subheader(f"Hello, Patient {patient_num}")
        st.markdown("""
                        Welcome to your health dashboard. 
                        Here you can find all the information about your health. Disclaimer, the information you can find here is based on a sample of 319,796 people thus the data might not represent the entire population""")

    random_features = pd.DataFrame({
            "BMICategory": [BMI],
            "Smoking": [smokingcat],
            "AlcoholDrinking": [alcohol],
            "Stroke": [strokecat],
            "PhysicalHealth": [physicalhealth],
            "MentalHealth": [0],
            "DiffWalking": [diffwalk],
            "Sex": [gender],
            "AgeCategory": [age],
            "Race": ["White"],
            "Diabetic": [diabeticcat],
            "PhysicalActivity": ["No"],
            "GenHealth": [genhealth],
            "SleepTime": [sleeptime],
            "Asthma": [asthma],
            "KidneyDisease": ["No"],
            "SkinCancer": ["No"]
        })

    random_features_dict = {
            "BMICategory": BMI,
            "Smoking": smokingcat,
            "AlcoholDrinking": alcohol,
            "Stroke": strokecat,
            "PhysicalHealth": physicalhealth,
            "MentalHealth": 0,
            "DiffWalking": diffwalk,
            "Sex": gender,
            "AgeCategory": age,
            "Race": "White",
            "Diabetic": diabeticcat,
            "PhysicalActivity": "No",
            "GenHealth": genhealth,
            "SleepTime": sleeptime,
            "Asthma": asthma,
            "KidneyDisease": "No",
            "SkinCancer": "No"
        }


    heart = load_dataset()



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
            st.markdown("<h2 style='text-align: center;' >Physical Activity </h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;' >" + physicalactivity +"<p>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<h2 style='text-align: center;' >Alcoholism </h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;' >" + alcohol +"<p>", unsafe_allow_html=True)
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
                "height": 200,
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
                "height": 200,
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


    # with col2.container(border=True):
    #     st.subheader("Your Heart Health Assistant")
    #     st.markdown("""<iframe src="https://vanherwegentim-chatbot-app-ci68bm.streamlit.app/?embed_options=disable_scrolling,show_padding,show_colored_line,show_toolbar,show_footer&embed=true" height="650" style="width: 100%; border: none;"></iframe>""", unsafe_allow_html=True)


    #############################
    # LangChain ChatGPT
    #############################
    def parse_counterfactuals(inference_data, cfe_raw):
        """
        Apply additonal boundary conditions
        # BMICategory should not go higher
        # Sleeptime should be higher or equal
        # if 0, it should stay as 0 for smoking and alcohol
        # if 1, it should stay as 1 for physical activity
        """
        bmicats = ["Underweight (BMI < 18.5)", "Normal weight (18.5 <= BMI < 25.0)", 
               "Overweight (25.0 <= BMI < 30.0)", "Obese (30.0 <= BMI < +Inf)"]
        # Assuming inference_data is a DataFrame and we're interested in the first row
        initial_bmi_category = inference_data["BMICategory"].iloc[0]
        initial_sleep_time = float(inference_data["SleepTime"].iloc[0])
        initial_smoking = inference_data["Smoking"].iloc[0]
        initial_alcohol = inference_data["AlcoholDrinking"].iloc[0]
        
        index_value1 = bmicats.index(initial_bmi_category)
        
        # To be dropped indices
        to_drop = []
        for ind, row in cfe_raw.iterrows():

            current_index = bmicats.index(row["BMICategory"])
            if initial_smoking == "No":
                row["Smoking"] = "No"
                
            if initial_alcohol == "No":
                row["AlcoholDrinking"] = "No"
            # If current BMI category is higher than the initial, mark for dropping
            if current_index > index_value1:
                to_drop.append(ind)
                continue  # Move to the next row
            
            # If SleepTime is not greater or equal to the initial, mark for dropping
            if float(row["SleepTime"]) < initial_sleep_time:
                to_drop.append(ind)
                continue
            
            
        
        # Drop the rows outside the loop
        cfe_raw.drop(labels=to_drop, axis=0, inplace=True)
        cfe_raw.drop_duplicates(subset=None, inplace=True)
        return cfe_raw






        #Creating counterfactuals
    def drop_not_wanted_features(df, features_to_drop, target_variable):
        '''
        Function to drop unwanted features
        '''
        for feature in features_to_drop:
            if feature in list(df.columns):
                df.drop(columns=feature, inplace=True)
                
        if target_variable in list(df.columns):
            df.drop(columns=target_variable, inplace=True)
        
        return df

    def load_counterfactuals():
        feature_ranges = {'SleepTime': (4,10), 'BMICategory':('Obese (30.0 <= BMI < +Inf)', 'Normal weight (18.5 <= BMI < 25.0)', 'Overweight (25.0 <= BMI < 30.0)')}
        exp = load("exp.joblib")
        e1 = exp.generate_counterfactuals(random_features, total_CFs=10, permitted_range=feature_ranges, desired_class="opposite", proximity_weight=1.5, diversity_weight=2.0, features_to_vary=["BMICategory", "Smoking", "SleepTime", "AlcoholDrinking", "DiffWalking","PhysicalActivity", "GenHealth"])
        cfe_json = json.loads(e1.to_json())        
        cfe_df = pd.DataFrame(cfe_json['cfs_list'][0],columns=cfe_json['feature_names_including_target'])
        return cfe_df
   

    class Droper(BaseEstimator, TransformerMixin):
        '''
        Adding a class for custom pipeline step
        '''
        def __init__(self, features_to_drop, target_variable):
                self.features_to_drop = features_to_drop
                self.target_variable = target_variable
                
        def fit(self, X, y):
                return self
            
        def transform(self, X):
            x = X.copy()
            return drop_not_wanted_features(x, self.features_to_drop, self.target_variable)
    
    @st.cache_data(persist=True)
    def pregenerate_counterfactual(random_features__):      
        max_retries = 50 # Maximum number of attempts
        retries = 0
        
        while retries < max_retries:
            # Attempt to load and parse counterfactuals
            cfs_list = load_counterfactuals()
            cfs_list = parse_counterfactuals(random_features, cfs_list)
            # Check if counterfactuals list is not empty
            if not cfs_list.empty:  # Assuming cfs_list is a DataFrame; adjust condition if it's a list
                cfs_list = cfs_list.iloc[0]
                cfs_list[13] = float(round(cfs_list[13]))
                counterfactuals_conditions = []
                feat = pd.DataFrame(random_features).T
                for i in range(14):
                    if cfs_list.iloc[i] != feat.iloc[i].iloc[0]:
                        counterfactuals_conditions.append({cfs_list.index[i]:(cfs_list[i])})
                return counterfactuals_conditions, cfs_list
            else:
                retries += 1

        
        return None  # or return an empty DataFrame/list as per your design
    
    @st.cache_data(persist=True)
    def generate_probability_prediction(cfs):
        cf_person = pd.DataFrame(cfs)
        cf_person.columns = cf_person.columns.astype(str)

        input_df = cf_person
        df = pd.concat([input_df, heart], axis=0)
        df = df.drop(columns=["HeartDisease"])
        cat_cols = ["BMICategory", "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking",
                    "Sex", "AgeCategory", "Race", "Diabetic", "PhysicalActivity",
                    "GenHealth", "Asthma", "KidneyDisease", "SkinCancer"]
        for cat_col in cat_cols:
            dummy_col = pd.get_dummies(df[cat_col], prefix=cat_col)
            df = pd.concat([df, dummy_col], axis=1)
            del df[cat_col]
        df = df[:1]
        df.fillna(0, inplace=True)
        prediction_prob = log_model.predict_proba(df)  
        return round(prediction_prob[0][1] * 100, 2)
    
    
    predicition_prob = generate_probability_prediction(random_features)
    counterfactual, cfs_list = pregenerate_counterfactual(random_features)
    prediction_prob_of_counterfactual = generate_probability_prediction(pd.DataFrame(cfs_list).T)

    

    
    @tool
    def generate_counterfactual():
        """This generates a heart disease counterfactual. The response is shown verbose.""" 
        newline = "\n"
        return f"""To potentially lower your heart disease risk, here are some changes that could be made based on a counterfactual scenario:
            {newline.join(f"{cf}" for cf in counterfactual)}
            These changes are hypothetical scenarios that suggest how certain lifestyle modifications could potentially reduce your heart disease risk. It's important to consult with healthcare professionals before making any significant changes to your lifestyle or health regimen.

            Would you like to see the predicted heart disease risk if these changes were implemented?"""


    @tool
    def prediction_of_heart_disease_risk_of_counterfactual():
        """Predicts the heart disease risk of a counterfactual. The response is shown verbose."""
        return prediction_prob_of_counterfactual
                


    @tool
    def predict_model() -> int:
        """Predicts the heart disease risk of a patient. The response is shown verbose."""
        return f"""Your heart disease risk is estimated to be {predicition_prob}%. Are you interested in learning more about how to manage your heart disease risk? Discover what actions you can take and see how your risk projection might change following certain lifestyle adjustments."""
        
    

    model = ChatOpenAI(model="gpt-4-0125-preview", temperature=0, api_key=st.secrets["OPENAI_API_KEY"], streaming=True)
    tools = [predict_model, generate_counterfactual, prediction_of_heart_disease_risk_of_counterfactual]
    ## add session state to fix memory issue
    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history",return_messages=True,)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        memory.load_memory_variables({})
    else:
        memory = st.session_state["memory"]

    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", "You are a helpful heart disease assistant"),
    #         MessagesPlaceholder("chat_history", optional=True),
    #         ("human", "{input}"),
    #         MessagesPlaceholder("agent_scratchpad"),
            
    #     ]
    # )
    prompt = hub.pull("hwchase17/openai-tools-agent")
    agent = create_openai_tools_agent(model, tools, prompt)
    agent_executor  = AgentExecutor(
    agent=agent, tools=tools, verbose=True

    )
    


    with col2.container(border=True, height=650):
        st.subheader("Your Heart Health Assistant")
        st.markdown("<p>Here you can ask any questions you have about your health disease risk. The AI will try to answer them to the best of its ability. For example, ask it what your risk of heart disease is.</p>", unsafe_allow_html=True)
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today? I can answer all your questions about your heart disease risk. Als je mij aanspreekt in het Nederlands, kan ik je ook in het Nederlands antwoorden."}]

        # Temporarily store the new message from chat input to add it after rendering the previous messages
        new_message_content = st.session_state.get("new_message_content", None)

        # Render existing messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):  
                st.markdown(message["content"])

        # Check if there's a new message to add
        if new_message_content:
            st.session_state.messages.append({"role": "user", "content": new_message_content})
            with st.chat_message("user"):
                st.markdown(new_message_content)
            with st.chat_message("assistant"):
                with st.spinner("Processing..."):
                    response = agent_executor.invoke({"input": new_message_content,
                                                      "chat_history": st.session_state.messages,})
                    st.write(response["output"])
                    # st.write_stream(get_response(new_message_content, ChatPromptTemplate.from_list(st.session_state.messages)))

            # Add the assistant's response to the chat history
            st.session_state.messages.append({"role": "assistant", "content": response["output"]})
            # Clear the temporary store to reset for the next input
            st.session_state.new_message_content = None

        # Place the chat input at the bottom. Upon submission, store the content temporarily and trigger a rerender
        new_message_content = st.chat_input("What is my heart disease risk?", key="new_chat_input")
        if new_message_content:
            st.session_state.new_message_content = new_message_content
            st.rerun()
      



if selected == "About":
    col1, col2, _ = st.columns([1, 2,1])
    with col2.container(border=True):
        st.subheader("Contact Information")
        st.markdown("""Tim Vanherwegen""")
        st.markdown("+32495197991")
        st.markdown("tim.vanherwegen.tv@gmail.com")

    with col1.container():
        st.image("images/handholding.png")