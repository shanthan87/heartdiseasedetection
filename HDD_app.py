import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from PIL import Image
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="HFD by KR",
    page_icon="*",
    initial_sidebar_state="expanded", )

col1, col2, col3 = st.beta_columns([1, 1, 1])

with col1:
    st.title('Heart')
    st.write('Random')

with col2:
    st.title('Failure')
    st.write('Forest')
    image = Image.open('heart.png')
    st.image(image, use_column_width=True)

with col3:
    st.title('Detection')
    st.write('Algorithm')

df = pd.read_csv('DataSet.csv')

X = df.iloc[:, 0:12].values
Y = df.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


def get_user_input():
    age = st.sidebar.text_input("Age", 0)
    anaemia = st.sidebar.selectbox('Do You have Anaemia? (0 = No , 1 = Yes)', ('0', '1'))
    creatinine_phosphokinase = st.sidebar.slider('Anaemia', 0, 8000, 4000)
    diabetes = st.sidebar.selectbox('Do You have Diabetes? (0 = No , 1 = Yes)', ('0', '1'))
    ejection_fraction = st.sidebar.slider('Ejection Fraction', 0, 100, 50)
    high_blood_pressure = st.sidebar.selectbox('Do You have HBP? (0 = No , 1 = Yes)', ('0', '1'))
    platelets = st.sidebar.slider('Platelets', 0, 900000, 450000)
    serum_creatinine = st.sidebar.slider('Serum Creatinine', 0.0, 10.0, 5.0)
    serum_sodium = st.sidebar.slider('Serum Sodium', 0, 150, 75)
    sex = st.sidebar.selectbox('Select your Gender? (0 = F , 1 = M)', ('0', '1'))
    smoking = st.sidebar.selectbox('Do You Smoke? (0 = No , 1 = Yes)', ('0', '1'))
    time = st.sidebar.slider('Time', 0, 300, 150)

    user_data = {'Age': age,
                 'Anaemia': anaemia,
                 'CP': creatinine_phosphokinase,
                 'Diabetes': diabetes,
                 'EF': ejection_fraction,
                 'HBP': high_blood_pressure,
                 'Platelets': platelets,
                 'SC': serum_creatinine,
                 'SS': serum_sodium,
                 'Sex': sex,
                 'Smoking': smoking,
                 'Time': time
                 }

    features = pd.DataFrame(user_data, index=[0])
    return features


user_input = get_user_input()

st.subheader('Current Data Input')
st.write(user_input)

col1, col2, col3 = st.beta_columns([1, 1, 1])

with col1:
    st.write('Anaemia (0 = No , 1 = Yes)')
    st.write('Diabetes (0 = No , 1 = Yes)')

with col2:
    st.write('HBP (0 = No , 1 = Yes)')

with col3:
    st.write('Smoke (0 = No , 1 = Yes)')
    st.write('Gender (0 = F , 1 = M)')

RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

st.title('Chances of Heart Diseases')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100) + '%')

prediction = RandomForestClassifier.predict(user_input)

st.title('Dataset')

d1 = df[(df["DEATH_EVENT"] == 0) & (df["sex"] == 1)]
d2 = df[(df["DEATH_EVENT"] == 1) & (df["sex"] == 1)]
d3 = df[(df["DEATH_EVENT"] == 0) & (df["sex"] == 0)]
d4 = df[(df["DEATH_EVENT"] == 1) & (df["sex"] == 0)]

label1 = ["Male", "Female"]
label2 = ['Male - Survived', 'Male - Died', "Female -  Survived", "Female - Died"]
values1 = [(len(d1) + len(d2)), (len(d3) + len(d4))]
values2 = [len(d1), len(d2), len(d3), len(d4)]

# Create subplots: use 'domain' type for Pie subplot
fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
fig.add_trace(go.Pie(labels=label1, values=values1, name="GENDER"),
              1, 1)
fig.add_trace(go.Pie(labels=label2, values=values2, name="GENDER VS DEATH"),
              1, 2)

# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.4, hoverinfo="label+percent")

fig.update_layout(
    title_text="GENDER  \
                   GENDER VS DEATH",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='GENDER', x=0.19, y=0.5, font_size=10, showarrow=False),
                 dict(text='GENDER VS DEATH', x=0.84, y=0.5, font_size=9, showarrow=False)],
    autosize=False, width=700, height=500, paper_bgcolor="white")

st.plotly_chart(fig)
