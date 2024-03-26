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
from sklearn.base import BaseEstimator, TransformerMixin
from streamlit_option_menu import option_menu
from helperfunctions import *
import math
import lightgbm as lgb



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
    
    if 'num' not in st.session_state:
        num=0
    else:
        num = st.session_state.num

    

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
    diffwalk = chosen_person[7]
    gender = chosen_person[8]
    age = chosen_person[9]
    diabeticcat = chosen_person[11]
    physicalactivity = chosen_person[12]
    genhealth = chosen_person[13]
    sleeptime = chosen_person[14]
    asthma = chosen_person[15]

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



    @st.cache_data(persist=True)
    def load_dataset() -> pd.DataFrame:
        # Assuming you have converted your dataset to Parquet format
        # and updated the DATASET_PATH to point to the .parquet file
        parquet_file_path = DATASET_PATH  # Update this to your Parquet file path
        heart_df = pd.read_parquet(parquet_file_path)
        return heart_df

    def user_input_features() -> pd.DataFrame:
        col1, col2 = col2cont.columns([1, 1])
        # race = col1cont.selectbox("Race", options=(race for race in heart.Race.unique()))
        sex = col1.selectbox("Sex", options=(sex for sex in heart.Sex.unique()),index=sex_to_numeric(gender) )
        age_category_order = [
            "18-24", "25-29", "30-34", "35-39", 
            "40-44", "45-49", "50-54", "55-59", 
            "60-64", "65-69", "70-74", "75-79", 
            "80 or older"
        ]
        
        age_cat = col1.selectbox("Age category",
                                    options=age_category_order, index=age_to_numeric(age))
        bmi_cat = col1.selectbox("BMI category",
                                    options=(bmi_cat for bmi_cat in heart.BMICategory.unique()), index=BMI_to_numeric(BMI))
        sleep_time = col2.number_input("How many hours on average do you sleep?", 0, 24,value=int(sleeptime))
        gen_health = col1.selectbox("How can you define your general health?",
                                        options=(gen_health for gen_health in heart.GenHealth.unique()), index=gen_health_to_numeric(genhealth))
        # phys_health = col1cont.number_input("For how many days during the past 30 days was"
        #                                       " your physical health not good?", 0, 30, 0)
        # # ment_health = col1cont.number_input("For how many days during the past 30 days was"
        #                                       " your mental health not good?", 0, 30, 0)
        # phys_act = col1cont.selectbox("Have you played any sports (running, biking, etc.)"
        #                                 " in the past month?", options=("No", "Yes"))
        smoking = col1.selectbox("Have you smoked at least 100 cigarettes in"
                                    " your entire life (approx. 5 packs)?)",
                                    options=("No", "Yes"), index=smoking_to_numeric(smokingcat))
        alcohol_drink = col2.selectbox("Do you have more than 14 drinks of alcohol (men)"
                                            " or more than 7 (women) in a week?", options=("No", "Yes"), index=alcohol_to_numeric(alcohol))
        stroke = col2.selectbox("Have you ever had a stroke?", options=("No", "Yes"), index=stroke_to_numeric(strokecat))
        diff_walk = col2.selectbox("Do you have difficulty walking"
                                        " or climbing stairs?", options=("No", "Yes"), index=diffwalk_to_numeric(diffwalk))
        diabetic = col2.selectbox("Do you have diabetes?",
                                        options=(diabetic for diabetic in heart.Diabetic.unique()), index=diabetic_to_numeric(diabeticcat))
        # asthma = col3.selectbox("Do you have asthma?", options=("No", "Yes"))
        # kid_dis = col1cont.selectbox("Do you have kidney disease?", options=("No", "Yes"))
        # skin_canc = col1cont.selectbox("Do you have skin cancer?", options=("No", "Yes"))
        features = pd.DataFrame({
            "BMICategory": [bmi_cat],
            "Smoking": [smoking],
            "AlcoholDrinking": [alcohol_drink],
            "Stroke": [stroke],
            "PhysicalHealth": [0],
            "MentalHealth": [0],
            "DiffWalking": [diff_walk],
            "Sex": [sex],
            "AgeCategory": [age_cat],
            "Race": ["White"],
            "Diabetic": [diabetic],
            "PhysicalActivity": ["No"],
            "GenHealth": [gen_health],
            "SleepTime": [sleep_time],
            "Asthma": ["No"],
            "KidneyDisease": ["No"],
            "SkinCancer": ["No"]
        })
        return features

    def chart_spec_pie(category_counts_dicts):
        return {
        "data": {
            "values": category_counts_dicts  # Use static list of dicts
        },
        "width": 200,  # Set the width of the chart
        "height": 150, # Set the height of the chart
        "mark": {"type": "arc", "innerRadius": 30},
        "encoding": {
            "theta": {"field": "Count", "type": "quantitative"},
            "color": {"field": "Category", "type": "nominal", "legend": {"title": "Categories"}}
        },
        }
        
    col1, col2 = st.columns([3,3])
    data = np.random.randn(10, 1)
    # contcol1 = col1.container()
    contcontcol1 = col1.container(border=True)
    contcontcol1.subheader(f"Hello, Patient {patient_num}")
    contcontcol1.markdown("""
                    Welcome to your health dashboard. 
                    Here you can find all the information about your health. Disclaimer, the information you can find here is based on a sample of 319,796 people thus the data might not represent the entire population""")
    contcol2 = col2.container(border=True)

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

    def generate_and_parse_counterfactuals(random_features):
        max_retries = 50 # Maximum number of attempts
        retries = 0
        
        while retries < max_retries:
            # Attempt to load and parse counterfactuals
            cfs_list = load_counterfactuals()
            cfs_list = parse_counterfactuals(random_features, cfs_list)
            
            # Check if counterfactuals list is not empty
            if not cfs_list.empty:  # Assuming cfs_list is a DataFrame; adjust condition if it's a list
                print("Successfully generated counterfactuals.")
                return cfs_list
            else:
                print(f"No counterfactuals found on attempt {retries + 1}. Retrying...")
                retries += 1
        
        print("Failed to generate non-empty counterfactuals after maximum retries.")
        return None  # or return an empty DataFrame/list as per your design

    # Use the function with your random_features DataFrame
    cfs_list = generate_and_parse_counterfactuals(random_features)
    

    
    
    with col2.container():
        cfs_list = cfs_list.iloc[0]
        # Define a list of conditions and messages for counterfactuals
        counterfactuals_conditions = [
            (cfs_list[0] != BMI, "BMI", "Reduce your weight", BMI, cfs_list[0]),
            (cfs_list[1] != smokingcat, "Smoking", "Stop smoking", None, None),
            # Add more conditions here
            (cfs_list[2] != alcohol, "Alcohol Drinking", "Stop Drinking alcohol", None, None),
            (cfs_list[6] != diffwalk, "Difficulty Walking", "Solve your difficulty walking", None, None),
            (cfs_list[11] != physicalactivity, "Physical Activity", "Get some more physical activity", None, None),
            (cfs_list[12] != genhealth, "General Health", "Improve your general health", genhealth, cfs_list[12]),
            # (cfs_list[13] != sleeptime, "Sleep Time", "Sleep more every night", str(sleeptime) + " hours", str(float(math.ceil(cfs_list[13]))) + " hours")
        ]
        st.subheader("Counterfactuals")

        st.markdown("<p>Here you can see what changes you can make to your lifestyle to receive a healthy prediction.</p>", unsafe_allow_html=True)
        
        
        first_shown = False  # Track whether the first counterfactual has been shown
        
        for condition, factor, action, current, target in counterfactuals_conditions:
            if condition:
                if not first_shown:
                    # Show the first counterfactual directly
                    with st.container(border=True):
                        st.markdown(f"<h5>Factor - {factor}</h5>", unsafe_allow_html=True)
                        st.markdown(f"Action - {action}")
                        if current and target:
                            st.markdown(f"{current} => {target}")
                    first_shown = True
                else:
                    # Use an expander for subsequent counterfactuals
                    with st.expander(f"Factor - {factor}"):
                        st.markdown(f"Action - {action}")
                        if current and target:
                            st.markdown(f"{current} => {target}")
        st.markdown("<p> Press the predict button below to see what effect the recommend changes would have on your predicted risk.</p>", unsafe_allow_html=True)
        colpred1, colpred2 = st.columns([1,1])
        if colpred1.button("Predict", use_container_width =True):
            heart = load_dataset()
            cf_person = pd.DataFrame(cfs_list).T
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
            log_model = pickle.load(open(LOG_MODEL_PATH, "rb"))

            prediction_prob = log_model.predict_proba(df)  
            delta_calculated = round(round(prediction_prob[0][1] * 100, 2) - float(st.session_state.prediction),2)

            colpred2.metric(label="Heart Disease Risk", value=str(round(prediction_prob[0][1] * 100, 2)) + " %", delta= str(delta_calculated) + " %", delta_color="inverse")
            colpred2.session_state.previous_state = round(prediction_prob[0][1] * 100, 2)








    # col2cont =  st.container(border=True)
    # #Prediction

    heart = load_dataset()


    # col2topcont = col2cont.container()
    # col2topcont1, col2topcont2 = col2topcont.columns([1,1])
    # submit = col2topcont1.button("Predict")

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

    # if submit:      
    #     delta_calculated = round(round(prediction_prob[0][1] * 100, 2) - st.session_state.previous_state,2)
    #     col2topcont2.metric(label="Heart Disease Risk", value=str(round(prediction_prob[0][1] * 100, 2)) + " %", delta= str(delta_calculated) + " %", delta_color="inverse")
    #     st.session_state.previous_state = round(prediction_prob[0][1] * 100, 2)

    #End Prediction

    
    st.session_state.prediction = str(round(prediction_prob[0][1] * 100, 2))
    st.session_state.prediction_bool = log_model.predict(df)




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
            st.markdown("<h2 style='text-align: center;' >Alcohol </h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;' >" + alcohol +"<p>", unsafe_allow_html=True)
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
            ("BMI", "Smoking", "Sleep Time", "General Health", "Alcohol Drinking", "Stroke", "Difficulty Walking", "Diabetes"),
            placeholder="Select a Graph to display",
        )

    # Assuming 'heart' is your DataFrame and 'col1' is a defined Streamlit container
    if option == "BMI":    
        with col1.container(border=True):
            bar_chart_spec = {
        "layer": [
            {
                "height": 200,
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
            st.markdown("<p>Smoking triples your heart disease risk.</p>", unsafe_allow_html=True)
            
            if smokingcat == "Yes":
                st.markdown(f"<p style='color: red;'>You are part of the {percentage}% that smokes</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='color: green;'>You are part of the {100 - percentage}% that doesn't smoke</p>", unsafe_allow_html=True)
            st.vega_lite_chart(pie_chart_spec, use_container_width=True)


        











#     #Start of the AI
# if selected == "Dashboard With AI":
#     load_dashboard_with_ai()


if selected == "About":
    col1, col2, _ = st.columns([1, 2,1])
    with col2.container(border=True):
        st.subheader("Contact Information")
        st.markdown("""Tim Vanherwegen""")
        st.markdown("+32495197991")
        st.markdown("tim.vanherwegen.tv@gmail.com")

    with col1.container():
        st.image("images/handholding.png")