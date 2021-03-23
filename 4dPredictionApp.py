import streamlit as st
import pandas as pd
import openpyxl
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
import numpy
import matplotlib.pyplot as plt
from PIL import Image

image = Image.open('logo.png')
st.image(image)
st.text("Please upload your file in the sidebar")

uploadFile = st.sidebar.file_uploader("Upload 4D excel file", type = [".xlsx"])
userToPredict = st.sidebar.number_input("Digit Sum to predict")
startAnalysis = st.sidebar.button("Start analysis")

if startAnalysis:
    dataset = pd.read_excel(uploadFile, engine='openpyxl')
    print(dataset.shape) # no of lines

    st.text("Here is the description of your dataset")

    # get description, write description
    st.write(dataset.describe())

    st.info("Here is a plot to see distributions and correlations")

    # scatter
    scatter_matrix(dataset, diagonal="hist", figsize = (8, 8));
    st.set_option('deprecation.showPyplotGlobalUse', False) # hide warning msg
    st.pyplot()

    # ============== ML ==============
    # get data
    x = dataset.loc[:, ["Digit Sum"]] 
    y = dataset.loc[:, ["4D Number"]] #labels

    # define model
    regressor = LinearRegression()

    # train model
    regressor.fit(x, y)

    # native test
    # print ("model fit is ok")

    xPredict = numpy.array([[
        userToPredict
    ]])

    # get prediction
    prediction = regressor.predict(xPredict)
    
    st.success("With Digit Sum " + str(xPredict[0][0]) 
                +  " the 4D Number is: " + str(prediction[0][0]))

    st.info("Here you can see your data (blue), model (red line) and prediction")
    
    # ============== LR ==============
    # create figure
    fix, ax = plt.subplots()

    # add lables to plots
    ax.set_xlabel("Digit Sum")
    ax.set_ylabel("4D Number")

    # dataset
    ax.scatter(x, y)

    # ploting the line
    ax.plot(x, regressor.intercept_[0] + regressor.coef_[0]*x, c = "r")

    # plot prediction
    ax.scatter(xPredict, prediction, c = "y", linewidth = 12)            

    # display plot
    st.pyplot(fix)


