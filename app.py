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

DATASET_PATH = "heart_2020_cleaned.csv"
LOG_MODEL_PATH = "logistic_regression.pkl"
#Gets random row from the dataset
total_rows= 319796
if "random_row_index" not in st.session_state:
    # If not, generate a new random row index and store it in the session state
    st.session_state.random_row_index = random.randint(0, total_rows - 1)

# Use the stored random row index to read the specific row from the CSV file
# This ensures the same row is used throughout the session
random_row_index = st.session_state.random_row_index
random_row = pd.read_csv(DATASET_PATH, skiprows=random_row_index, nrows=1)


BMI = random_row.iloc[0,1]
smokingcat = random_row.iloc[0,2]
alcohol = random_row.iloc[0,3]
strokecat = random_row.iloc[0,4]
physicalhealth = random_row.iloc[0,5]
diffwalk = random_row.iloc[0,7]
gender = random_row.iloc[0,8]
age = random_row.iloc[0,9]
diabeticcat = random_row.iloc[0,11]
genhealth = random_row.iloc[0,13]
sleeptime = random_row.iloc[0,14]
asthma = random_row.iloc[0,15]


@st.cache_data(persist=True)
def load_dataset() -> pd.DataFrame:
    heart_df = pl.read_csv(DATASET_PATH)
    heart_df = heart_df.to_pandas()
    heart_df = pd.DataFrame(np.sort(heart_df.values, axis=0),
                            index=heart_df.index,
                            columns=heart_df.columns)
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
# st.set_page_config(
#     page_title="Heart Disease Prediction App",
#     page_icon="images/heart-fav.png"
# )
st.title("Heart Disease Prediction")

col1, col2, col3 = st.columns([3,3,3])
data = np.random.randn(10, 1)
# contcol1 = col1.container()
contcontcol1 = col1.container(border=True)
contcontcol1.subheader("Hello, Jane Doe")
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
    st.session_state.previous_state = 0
if submit:      
    delta_calculated = round(round(prediction_prob[0][1] * 100, 2) - st.session_state.previous_state,2)
    col2topcont2.metric(label="Heart Disease Risk", value=str(round(prediction_prob[0][1] * 100, 2)) + " %", delta= str(delta_calculated) + " %", delta_color="inverse")
    st.session_state.previous_state = round(prediction_prob[0][1] * 100, 2)

#End Prediction

if "prediction" not in st.session_state:
    st.session_state.prediction = str(round(prediction_prob[0][1] * 100, 2))
    st.session_state.prediction_bool = log_model.predict(df)

# komma = int(round(float(st.session_state.prediction)))/100
# with contcol2:
#     st.write(komma)
#     sv.gauge(komma,sFix="%", gSize="SML", gTheme="white")

# with contcol2:
#     fig = go.Figure(go.Indicator(
#     mode="gauge+number",
#     value=float(st.session_state.prediction),
#     domain={'x': [0, 1], 'y': [0, 1]},
#     title={'text': "Heart Disease Risk"},
#     gauge={
#         'axis': {'range': [None, 100]},  # Set the range to be always up to 100
#         'steps': [
#             {'range': [0, 20], 'color': "green"},
#             {'range': [20, 40], 'color': "yellow"},
#             {'range': [40, 100], 'color': "red"}
#         ],
#         'bar': {'color': "black"},
#     },
#     number={
#             'valueformat': "2.2f"  # Format the value to two decimal places
#         },

# ))
#     fig.update_layout(
#     margin={'t': 50, 'b': 20, 'l': 20, 'r': 20},  # Reduce margin to make the figure more compact
#     height=200,  # Set the height of the figure
#     width=250    # Set the width of the figure
# )

#     st.plotly_chart(fig, use_container_width=True)
# names=['Heart Disease Risk', '']
# size_of_groups=[float(st.session_state.prediction),100-float(st.session_state.prediction)]
# plt.pie(size_of_groups,labels=names,colors=['red','white'])
# fig = plt.figure()
# fig.patch.set_facecolor('#0E1117')  # Set the background color to "#0E1117"
# plt.rcParams['text.color'] = 'white'

# my_circle=plt.Circle( (0,0), 0.7, color='#0E1117')  # Set the background color to "#0E1117"
# p=plt.gcf()
# p.gca().add_artist(my_circle)
    # library 
# Data

# Values for the pie chart
# values = [float(st.session_state.prediction), 100 - float(st.session_state.prediction)]
# labels = ['Yes', 'No']

# # Create the pie chart
# fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
# fig.update_layout(width=400, height=300)

# # Display the pie chart in the Streamlit app
# with contcol2:
#     st.plotly_chart(fig)

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

    # st.write("Your blood pressure is 120/80, which is abnormaly high")
    # data = np.random.randn(10, 1)
    # chart_data = pd.DataFrame(np.random.randn(20, 2), columns=["x", "y"])
    # st.line_chart(chart_data)


# def plot_gen_health_distribution(heart_df, user_value):
#     # Create a figure with a dark background
#     # plt.style.use('dark_background')
#     fig, ax = plt.subplots(facecolor='#0E1117')
    
#     # Histogram for the GenHealth column with a lighter color for visibility
#     ax.hist(heart_df['BMICategory'], bins=4, alpha=0.7, color='white', label='BMI Distribution')
    
#     # Marker for the user's GenHealth value
#     ax.axvline(x=user_value, color='red', linestyle='--', linewidth=2, label=f'Your BMI ({user_value})')
    
#     # Set labels and title with a lighter color
#     ax.set_xlabel('Physical Health', color='white')
#     ax.set_ylabel('Number Of People', color='white')
#     ax.set_title('Physical Health Distribution with Your Value', color='white')
    
#     # Change tick colors
#     for axis in ['top', 'bottom', 'left', 'right']:
#         ax.spines[axis].set_color('white')
#     ax.tick_params(axis='x', colors='white')
#     ax.tick_params(axis='y', colors='white')
    
#     # Customize legend to be more visible in dark theme
#     legend = ax.legend()
#     for text in legend.get_texts():
#         text.set_color('white')
    
#     # Set the figure's face color to match Streamlit's dark theme background
#     # Set axes face color to match the overall dark theme
#     ax.set_facecolor('#0E1117')
    
#     return fig


# with col1.container(border=True):
#     st.subheader("Physical Health Distribution")
#     st.write(physicalhealth)
#     fig = plot_gen_health_distribution(heart, BMI)
#     st.pyplot(fig)


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
                    "aggregate": "count",
                    "type": "quantitative",
                    "title": "Number of People"
                }
            }
        },
        {
            "mark": {"type": "rule", "tooltip": {"content": BMI}},
            "encoding": {
                "x": {
                    "field": "BMICategory",  # The field must match the one used in the bar chart
                    "type": "nominal",
                    # Specify the category you want to highlight. This assumes you have a way to set it.
                    "datum": BMI  # Example: Highlighting the "Overweight" category
                },
                "color": {
                    "value": "red"  # Color of the rule
                },
                "size": {
                    "value": 2  # Width of the rule line
                }
            }
        }
    ]
}



# Assuming 'heart' is your DataFrame and 'col1' is a defined Streamlit container
with col1.container(border=True):
    st.markdown(f"The red line is your BMI category: :red[{BMI}]")

    st.vega_lite_chart(heart, bar_chart_spec,use_container_width=True)


smokingcontainer = col1.container(border=True)
col1col1, col1col2 = smokingcontainer.columns([1,1])

category_counts = heart['Smoking'].value_counts().reset_index()
category_counts.columns = ['Category', 'Count']
category_counts_dicts = category_counts.to_dict('records')
pie_chart_spec = {
    "data": {
        "values": category_counts_dicts  # Convert the DataFrame to a list of dicts
    },
    "width": 200,   # Set the width of the chart
    "height": 200, # Set the height of the chart
    "mark": "arc",
    "encoding": {
        "theta": {"field": "Count", "type": "quantitative"},
        "color": {"field": "Category", "type": "nominal", "legend": {"title": "Categories"}}
    },
}
no_smoking = category_counts_dicts[0]['Count']
yes_smoking = category_counts_dicts[1]['Count']
percentage = round((yes_smoking / (yes_smoking + no_smoking)) * 100, 2)
# Use Streamlit's vega_lite_chart method to render the pie chart

with smokingcontainer.container():
    st.subheader("Smoking")
    st.markdown("<p>Smoking triples your heart disease risk.</p>", unsafe_allow_html=True)
    
    if smokingcat == "Yes":
        st.markdown(f"<p style='color: red;'>You are part of the {percentage}% that smokes</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='color: green;'>You are part of the {100 - percentage}% that doesn't smoke</p>", unsafe_allow_html=True)
    st.vega_lite_chart(pie_chart_spec, use_container_width=True)


    



# with col2.container():
#     st.subheader("Your Risk Over Time")
# with col2.container(border=True):
#     chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
#     st.bar_chart(chart_data)
# with col2.container(border=True):
#     st.subheader("Early symptoms of a heart attack")
#     st.markdown("<h4>Chest pain or discomfort </h4>", unsafe_allow_html=True)
#     st.markdown("Most heart attacks involve discomfort in the center or left side of the chest that lasts for more than a few minutes or that goes away and comes back. The discomfort can feel like uncomfortable pressure, squeezing, fullness, or pain.")
#     st.markdown("<h4>Feeling weak, light-headed, or faint</h4>", unsafe_allow_html=True)
#     st.markdown("<h4>Pain or discomfort in one or both arms or shoulders.</h4>", unsafe_allow_html=True)
#     st.markdown("<h4>Shortness of breath</h4>", unsafe_allow_html=True)



with col3.container(border=True):
    st.title("ChatGPT-like clone")

    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
# def chat_content():
#     st.session_state['contents'].append(st.session_state.content)

# if 'contents' not in st.session_state:
#     st.session_state['contents'] = []
#     border = False
# else:
#     border = True

# with contcol3:
#     with st.container(border=border):
#         with st.container():
#             st.chat_input(key='content', on_submit=chat_content) 
#             button_b_pos = "0rem"
#             button_css = float_css_helper(width="2.2rem", bottom=button_b_pos, transition=0)
#             float_parent(css=button_css)
#         if content:=st.session_state.content:
#             with st.chat_message(name='robot'):
#                 for c in st.session_state.contents:
#                     st.write(c)
                    
