import streamlit as st
import pandas as pd
import pickle
def get_features():
    st.title('Logistic Regression Deployment')
    st.sidebar.title("Enter Parameters Values")

    Pclass = st.sidebar.radio('PClass', [1, 2, 3])

    # Convert string 'True'/'False' to actual boolean, then to int
    Embarked_Q = int(st.sidebar.radio('Embarked_Q', ['True', 'False']) == 'True')
    Embarked_S = int(st.sidebar.radio('Embarked_S', ['True', 'False']) == 'True')
    Sex_male = int(st.sidebar.radio('IsMale', ['True', 'False']) == 'True')

    Age = st.sidebar.slider('Age', min_value=1, max_value=100)
    SibSp = int(st.sidebar.selectbox('SibSp', ['0', '1', '2', '3', '4', '5', '8']))
    Parch = int(st.sidebar.selectbox("Parch", ['0', '1', '2', '3', '4', '5', '6']))

    data = {
        'Pclass': Pclass,
        'Age': Age,
        'SibSp': SibSp,
        'Parch': Parch,
        'Embarked_Q': Embarked_Q,
        'Embarked_S': Embarked_S,
        'Sex_male': Sex_male
    }

    features = pd.DataFrame(data, index=[0])
    return features


xvals=get_features()
if st.sidebar.button('Submit'):
    st.write(xvals)
    loaded_model=pickle.load(open('clf.pkl','rb'))
    result=loaded_model.predict(xvals)
    st.write(result)
