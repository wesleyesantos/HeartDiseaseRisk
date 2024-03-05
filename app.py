import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import pickle
import sklearn
import plotly.graph_objs as go
import random
# import streamviz as sv
import matplotlib.pyplot as plt

from helperfunctions import *

# from streamlit_float import *

from openai import OpenAI


st.set_page_config(layout="wide")
# float_init(theme=True, include_unstable_primary=False)

DATASET_PATH = "heart_2020_cleaned.parquet"
LOG_MODEL_PATH = "logistic_regression.pkl"
#Gets random row from the dataset
total_rows= 319796
if "random_row_index" not in st.session_state:
    # If not, generate a new random row index and store it in the session state
    st.session_state.random_row_index = random.randint(0, total_rows - 1)


random_person_1 = ["128868","Obese (30.0 <= BMI < +Inf)","Yes","No","No","12.0","10.0","Yes","Male","50-54","White","Yes","No","Poor",6.0,"No","No","No"]
random_person_2 = ["222411","Overweight (25.0 <= BMI < 30.0)","No","No","No","0.0","0.0","Yes","Male","75-79","White","No","No","Very good",7.0,"No","No","No"]
random_person_3 = ["222428","Overweight (25.0 <= BMI < 30.0)","Yes","No","No","0.0","1.0","No","Female","70-74","White","No","No","Very good",7.0,"No","No","No"]
random_person_4 = ["8486","Overweight (25.0 <= BMI < 30.0)","Yes","No","No","15.0","4.0","Yes","Male","35-39","White","No","No","Good",6.0,"No","No","No"]
random_person_list = [random_person_1, random_person_2, random_person_3, random_person_4]


if "chosen_person" not in st.session_state:
    random_number = random.randint(0, 3)
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
genhealth = chosen_person[13]
sleeptime = chosen_person[14]
asthma = chosen_person[15]



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
    age_cat = col1.selectbox("Age category",
                                   options=(age_cat for age_cat in heart.AgeCategory.unique()), index=age_to_numeric(age))
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
        "PhysicalHealth": ["0"],
        "MentalHealth": ["0"],
        "SleepTime": [sleep_time],
        "BMICategory": [bmi_cat],
        "Smoking": [smoking],
        "AlcoholDrinking": [alcohol_drink],
        "Stroke": [stroke],
        "DiffWalking": [diff_walk],
        "Sex": [sex],
        "AgeCategory": [age_cat],
        "Race": ["White"],
        "Diabetic": [diabetic],
        "PhysicalActivity": ["No"],
        "GenHealth": [gen_health],
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
    "height": 200, # Set the height of the chart
    "mark": "arc",
    "encoding": {
        "theta": {"field": "Count", "type": "quantitative"},
        "color": {"field": "Category", "type": "nominal", "legend": {"title": "Categories"}}
    },
    }
    
# st.set_page_config(
#     page_title="Heart Disease Prediction App",
#     page_icon="images/heart-fav.png"
# )
st.title("Heart Disease Prediction")

col1, col2, col3 = st.columns([3,3,3])
data = np.random.randn(10, 1)
# contcol1 = col1.container()
contcontcol1 = col1.container(border=True)
contcontcol1.subheader(f"Hello, Patient {patient_num}")
contcontcol1.markdown("""
                  Welcome to your health dashboard. 
                  Here you can find all the information about your health.""")
contcol2 = col2.container(border=True)


col2.subheader("What If?")


col2cont =  col2.container(border=True)
#Prediction

heart = load_dataset()


col2topcont = col2cont.container()
col2topcont1, col2topcont2 = col2topcont.columns([1,1])
submit = col2topcont1.button("Predict")
col2cont.markdown("<p>Here you can see how much your predicted heart risk would change if you would make some lifestyle changes. Press the Predict button after changing a variable.</p>", unsafe_allow_html=True)
input_df = user_input_features()
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
    st.session_state.previous_state = 0
if submit:      
    delta_calculated = round(round(prediction_prob[0][1] * 100, 2) - st.session_state.previous_state,2)
    col2topcont2.metric(label="Heart Disease Risk", value=str(round(prediction_prob[0][1] * 100, 2)) + " %", delta= str(delta_calculated) + " %", delta_color="inverse")
    st.session_state.previous_state = round(prediction_prob[0][1] * 100, 2)

#End Prediction

if "prediction" not in st.session_state:
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
st.sidebar.image('pictures/stock_placeholder.jpg', width=100)
sidecont1, sidecont2 = st.sidebar.columns([3, 3])

with sidecont1:
    with st.container(border=True):
        st.markdown("<h2 style='text-align: center;' >Age </h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;' >" + age +"<p>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("<h2 style='text-align: center;' >BMI </h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;' >" + BMI +"<p>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("<h2 style='text-align: center;' >Smoking </h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;' >" + smokingcat +"<p>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("<h2 style='text-align: center;' >Stroke </h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;' >" + strokecat +"<p>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("<h2 style='text-align: center;' >Asthma </h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;' >" + asthma +"<p>", unsafe_allow_html=True)

with sidecont2:
    with st.container(border=True):
        st.markdown("<h2 style='text-align: center;' >Gender </h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;' >" + gender +"<p>", unsafe_allow_html=True)
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
        st.markdown("<h2 style='text-align: center;' >General Health </h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;' >" + genhealth +"<p>", unsafe_allow_html=True)






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


with col3.container(border=True):
    st.title("HeartGPT")
    st.markdown("<p>Here you can ask any questions you have about your health. The AI will try to answer them to the best of its ability.</p>", unsafe_allow_html=True)
    st.markdown("""<iframe src="https://vanherwegentim-chatbot-app-ci68bm.streamlit.app/?embed_options=disable_scrolling,show_padding,show_colored_line,show_toolbar,show_footer&embed=true" height="600" style="width: 100%; border: none;"></iframe>""", unsafe_allow_html=True)

    