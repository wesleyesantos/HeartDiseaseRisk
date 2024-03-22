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
    col2.subheader("Welcome to the Heart Health Advisor")
    col2.markdown("""At HeartBeat, we're dedicated to empowering you on your journey to heart health.
                   Our user-friendly app combines cutting-edge technology with personalized insights to help you understand and manage your heart disease risk. 
                  From clear explanations to easily understandable advice, we hope to assist you in your journey towards a healthy future.""")
    with col1.container():
        st.image("images/handholding.png")

if selected == "Dashboard":

    
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
    colcol1, colcol2 = col1.columns([3,2])
    contcontcol1 = colcol1.container(border=True)
    with col1.container(border=True):
        st.subheader(f"Hello, Patient {patient_num}")
        st.markdown("""
                        Welcome to your health dashboard. 
                        Here you can find all the information about your health. Disclaimer, the information you can find here is based on a sample of 319,796 people thus the data might not represent the entire population""")

    contcol2 = colcol2.container(border=True)
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




    input_df = random_features
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
    log_model = pickle.load(open(LOG_MODEL_PATH, "rb"))

    prediction_prob = log_model.predict_proba(df)  

    if "previous_state" not in st.session_state:
        #Change this to predicted
        st.session_state.previous_state = round(prediction_prob[0][1] * 100, 2)



    #End Prediction


    st.session_state.prediction = str(round(prediction_prob[0][1] * 100, 2))
    st.session_state.prediction_bool = log_model.predict(df)



    # contcol2.markdown("<p style='text-align: center;' > Your calculated risk is</p>", unsafe_allow_html=True)
    # if(st.session_state.prediction_bool == 0):
    #     contcol2.markdown("<h1 style='text-align:center;font-size:3rem; padding:0rem; color:green;'>" + st.session_state.prediction + "%</h1>", unsafe_allow_html=True)
    #     contcol2.markdown("<p style='text-align: center;' >Considered Healthy</p>", unsafe_allow_html=True)
    # else:
    #     contcol2.markdown("<h1 style='text-align:center;font-size:3rem; padding:0rem; color:red;'>" + st.session_state.prediction + "%</h1>", unsafe_allow_html=True)
    #     contcol2.markdown("<p style='text-align: center;' >Considered Unhealthy</p>", unsafe_allow_html=True)


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
    #     st.markdown("<p>Here you can ask any questions you have about your health. The AI will try to answer them to the best of its ability.</p>", unsafe_allow_html=True)
    #     st.markdown("""<iframe src="https://vanherwegentim-chatbot-app-ci68bm.streamlit.app/?embed_options=disable_scrolling,show_padding,show_colored_line,show_toolbar,show_footer&embed=true" height="650" style="width: 100%; border: none;"></iframe>""", unsafe_allow_html=True)


    #############################
    # LangChain ChatGPT
    #############################

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
    class Droper(BaseEstimator, TransformerMixin):
        '''
        Adding a clasws for custom pipeline step
        '''
        def __init__(self, features_to_drop, target_variable):
                self.features_to_drop = features_to_drop
                self.target_variable = target_variable
                
        def fit(self, X, y):
                return self
            
        def transform(self, X):
            x = X.copy()
            return drop_not_wanted_features(x, self.features_to_drop, self.target_variable)


    # class MLModel(BaseModel):
    #     BMICategory: str = "Normal weight (18.5 <= BMI < 25.0)"  
    #     Smoking: str = "Yes"
    #     AlcoholDrinking: str = "No"
    #     Stroke: str = "Yes"
    #     PhysicalHealth: float = 12.0
    #     MentalHealth: float = 10.0
    #     DiffWalking: str = "Yes"
    #     Sex: str = "Male"
    #     AgeCategory: str = "50-54"
    #     Race: str = "White"
    #     Diabetic: str = "Yes"
    #     PhysicalActivity: str = "No"
    #     GenHealth: str = "Excellent"
    #     SleepTime: float = 8.0
    #     Asthma: str = "No"
    #     KidneyDisease: str = "No"
    #     SkinCancer: str = "No"
    class MLModel(BaseModel):
        BMICategory: str = "Normal weight (18.5 <= BMI < 25.0)"  
        Smoking: str = "Yes"
        AlcoholDrinking: str = "No"
        Stroke: str = "Yes"
        PhysicalHealth: float = 12.0
        MentalHealth: float = 10.0
        DiffWalking: str = "Yes"
        Sex: str = "Male"
        AgeCategory: str = "50-54"
        Race: str = "White"
        Diabetic: str = "Yes"
        PhysicalActivity: str = "No"
        GenHealth: str = "Excellent"
        SleepTime: float = 8.0
        Asthma: str = "No"
        KidneyDisease: str = "No"
        SkinCancer: str = "No"


    @tool
    def call_cfs_generator(random_features) -> pd.DataFrame:
        """This returns counterfactuals, the reason why people have heart disease risk. Only use this when they request why they have a certain risk."""    
        random_features = pd.DataFrame(random_features, index=[0])
        
        
        feature_ranges = {'SleepTime': (4,10), 'BMICategory':('Obese (30.0 <= BMI < +Inf)', 'Normal weight (18.5 <= BMI < 25.0)', 'Overweight (25.0 <= BMI < 30.0)')}
        exp = joblib.load("exp.joblib")
        e1 = exp.generate_counterfactuals(random_features, total_CFs=1, permitted_range=feature_ranges, desired_class="opposite", proximity_weight=1.5, diversity_weight=2.0, features_to_vary=["BMICategory", "Smoking", "SleepTime", "AlcoholDrinking", "DiffWalking","PhysicalActivity", "GenHealth"])
        cfe_json = json.loads(e1.to_json())        
        cfe_df = pd.DataFrame(cfe_json['cfs_list'][0],columns=cfe_json['feature_names_including_target'])
        
        return cfe_df


    @tool
    def predict_model(new_value: MLModel) -> int:
        """This calculates their heart disease risk, Only use this when they request their heart disease risk"""
        model = pickle.load(open("logistic_regression.pkl", "rb"))
        
        # Convert MLModel instance to a dictionary, then to a DataFrame
        new_value_dict = new_value.dict()
        # new_value_df = pd.DataFrame([new_value_dict])  # Convert dict to DataFrame correctly
        heart = load_dataset()
        cf_person = pd.DataFrame(new_value_dict, index=[0])
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
        y_pred = model.predict_proba(df)[0][1]*100
        return y_pred

    model = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0, api_key=st.secrets["OPENAI_API_KEY"])
    # random_features_dict = {
    #         "BMICategory": "Obese (30.0 <= BMI < +Inf)",
    #         "Smoking": "Yes",
    #         "AlcoholDrinking": "No",
    #         "Stroke": "No",
    #         "PhysicalHealth": 12.0,
    #         "MentalHealth": 10.0,
    #         "DiffWalking": "Yes",
    #         "Sex": "Male",
    #         "AgeCategory": "50-54",
    #         "Race": "White",
    #         "Diabetic": "Yes",
    #         "PhysicalActivity": "No",
    #         "GenHealth": "Poor",
    #         "SleepTime": 6.0,
    #         "Asthma": "No",
    #         "KidneyDisease": "No",
    #         "SkinCancer": "No"
    #     }
    initial_data = json.dumps(random_features_dict)
    initial_data

    tools = [predict_model, call_cfs_generator]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory.load_memory_variables({})
    
    prompt = hub.pull("hwchase17/openai-tools-agent")
    agent = create_openai_tools_agent(model, tools, prompt)
    agent_executor  = AgentExecutor(
    agent=agent, tools=tools, verbose=True, memory=memory
    )


    # st.write(agent_executor.invoke(
    # {
    #     "input": f": What is my heart disease risk?: {initial_data}?"
    # }
    # ))
    with col2.container(border=True):
        st.subheader("Your Heart Health Assistant")
        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = "gpt-3.5-turbo"

        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("What is my heart disease risk?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                response = agent_executor.invoke({"input": prompt + f": {initial_data}?"})
                # stream = client.chat.completions.create(
                #     model=st.session_state["openai_model"],
                #     messages=[
                #         {"role": m["role"], "content": m["content"]}
                #         for m in st.session_state.messages
                #     ],
                #     stream=True,
                # )
                st.write(response["output"])
            st.session_state.messages.append({"role": "assistant", "content": response["output"]})

