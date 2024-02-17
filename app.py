import streamlit as st
import numpy as np
from streamlit_float import *
import pandas as pd

st.set_page_config(layout="wide")
float_init(theme=True, include_unstable_primary=False)

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

# Using object notation
st.sidebar.image('pictures/stock_placeholder.jpg', width=100)
sidecont1, sidecont2, sidecont3 = st.sidebar.columns([3, 3, 3])
sidecont1 = st.sidebar.container(border=True)
sidecont2 = st.sidebar.container(border=True)
sidecont3 = st.sidebar.container(border=True)

sidecont1.markdown("<h2 style='text-align: center;' >BloodPressure </h2>", unsafe_allow_html=True)
sidecont1.markdown("<p style='text-align: center;' >120/80 <p>", unsafe_allow_html=True)
sidecont2.markdown("<h2 style='text-align: center;' >Cholesterol </h2>", unsafe_allow_html=True)
sidecont2.markdown("<p style='text-align: center;' >220 <p>", unsafe_allow_html=True)
sidecont3.markdown("<h2 style='text-align: center;' >Gender </h2>", unsafe_allow_html=True)
sidecont3.markdown("<p style='text-align: center;' >Female <p>", unsafe_allow_html=True)


col1, col2, col3 = st.columns([3,3,3])
data = np.random.randn(10, 1)
# contcol1 = col1.container()
contcontcol1 = col1.container(border=True)
contcontcol1.subheader("Hello, Jane Doe")
contcontcol1.markdown("""
                  Welcome to your health dashboard. 
                  Here you can find all the information about your health.""")
with col1.container():
    st.subheader("Your results")
with col1.container(border=True):
    data = np.random.randn(10, 1)
    chart_data = pd.DataFrame(np.random.randn(20, 2), columns=["x", "y"])
    st.line_chart(chart_data)

smokingcontainer = col1.container(border=True)
col1col1, col1col2 = smokingcontainer.columns([1,1])

with col1col1.container():
    st.subheader("Smoking")
    st.markdown("<p>Smoking doubles your heart disease risk</p>", unsafe_allow_html=True)
with col1col2.container():
    st.markdown("<p style='text-align: center;' ></p>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align:center;font-size:2rem; padding:0rem;'>50 %</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;' >of the current population smokes</p>", unsafe_allow_html=True)

    


contcol2 = col2.container(border=True)
contcol2.markdown("<p style='text-align: center;' > Your calculated risk is</p>", unsafe_allow_html=True)
contcol2.markdown("<h1 style='text-align:center;font-size:3rem; padding:0rem;'>50 %</h1>", unsafe_allow_html=True)
contcol2.markdown("<p style='text-align: center;' >Considered quite high</p>", unsafe_allow_html=True)

with col2.container():
    st.subheader("Your Risk Over Time")
with col2.container(border=True):
    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
    st.bar_chart(chart_data)
with col2.container(border=True):
    st.subheader("Early symptoms of a heart attack")
    st.markdown("<h4>Chest pain or discomfort </h4>", unsafe_allow_html=True)
    st.markdown("Most heart attacks involve discomfort in the center or left side of the chest that lasts for more than a few minutes or that goes away and comes back. The discomfort can feel like uncomfortable pressure, squeezing, fullness, or pain.")
    st.markdown("<h4>Feeling weak, light-headed, or faint</h4>", unsafe_allow_html=True)
    st.markdown("<h4>Pain or discomfort in one or both arms or shoulders.</h4>", unsafe_allow_html=True)
    st.markdown("<h4>Shortness of breath</h4>", unsafe_allow_html=True)


# with col3.chat_message("user"):
#     col3.write("Hello 👋")

# with col3.chat_message("assistant"):
#     col3.write("Hello human")
#     col3.bar_chart(np.random.randn(30, 3))

contcol3 = col3.container(border=True)


def chat_content():
    st.session_state['contents'].append(st.session_state.content)

if 'contents' not in st.session_state:
    st.session_state['contents'] = []
    border = False
else:
    border = True

with contcol3:
    with st.container(border=border):
        with st.container():
            st.chat_input(key='content', on_submit=chat_content) 
            button_b_pos = "0rem"
            button_css = float_css_helper(width="2.2rem", bottom=button_b_pos, transition=0)
            float_parent(css=button_css)
        if content:=st.session_state.content:
            with st.chat_message(name='robot'):
                for c in st.session_state.contents:
                    st.write(c)
                    
colrow12, colrow22, colrow32 = st.columns([3,3,3])
data = np.random.randn(10, 1)
# with colrow12.container(height=400,border=True):
#     chart_data = pd.DataFrame(np.random.randn(20, 2), columns=["x", "y"])

#     st.line_chart(chart_data)

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat messages from history on app rerun


# # React to user input
# if prompt := col3.chat_input("What is up?"):
#     # Display user message in chat message container
#     with col3.chat_message("user"):
#         col3.markdown(prompt)
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     response = f"Echo: {prompt}"
#     # Display assistant response in chat message container
#     with col3.chat_message("assistant"):
#         col3.markdown(response)
#     # Add assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": response})

# for message in st.session_state.messages:
#     with col3.chat_message(message["role"]):
#         col3.markdown(message["content"])