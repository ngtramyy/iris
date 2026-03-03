import streamlit as st
import pickle
from sklearn.datasets import load_iris
 
iris = load_iris()

# Load the trained model
clf = pickle.load(open('model.pkl', 'rb'))

#sidebar for user input
st.sidebar.title('Iris Classifier')
sepal_length = st.sidebar.slider('Sepal Length', 4.0,8.0,5.0)
sepal_width = st.sidebar.slider('Sepal Width', 2.0,4.5,3.0)
petal_length = st.sidebar.slider('Petal Length', 1.0,7.0,4.0)
petal_width = st.sidebar.slider('Petal Width', 0.1,2.5,1.3)

#make prediction
prediction = clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])

#display results
st.write('Prediction:')
st.write(iris.target_names[prediction][0])

