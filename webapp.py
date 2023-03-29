import pandas as pd
import streamlit as st
st.title('GET THE GROSS SALE')
def file_uploader():
    file = st.sidebar.file_uploader("Upload a file", type=["pdf", "txt","csv"])
    if file is not None:
        df= pd.read_csv(file)
        df
        df1=df.dropna()
        import sklearn
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        import matplotlib.pyplot as plt
        from sklearn import metrics
        #Setting the value for X and Y
        x = df1[['sales rank','Book_ratings_count','publisher revenue']]
        y = df1['gross sales']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)
        slr= LinearRegression()  #simple linear regression
        slr.fit(x_train, y_train)
        print('Intercept: ', slr.intercept_)
        list(zip(x ,slr.coef_))
        #Prediction of Test and Training set result  
        y_pred_slr= slr.predict(x_test)  
        x_pred_slr= slr.predict(x_train) 
        #Actual value and the predicted value
        slr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_slr})
        slr_diff
        rank = st.number_input("enter sales rank", step=1)
        rating = st.number_input("enter book rating count", step=1)
        revenue = st.number_input("enter publisher revenue($)", step=1)
        k=slr.predict([[rank,rating,revenue]])
        st.write("the gross sale($) is:")
        st.write(k)
        # print the R-squared value for the model
        from sklearn.metrics import accuracy_score
        j=slr.score(x,y)*100
        st.write("accuracy of the model is:")
        st.write(j)
        #st.write("File contents:")
        #st.write(content)

# Call the function to display the file uploader
file_uploader()