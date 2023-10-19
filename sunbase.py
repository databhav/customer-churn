# loading required libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(layout="wide")
st.header("Customer Churn Project:")
st.write("The following is the model performance of customer churn prediction. The deployment only shows the model performance as the data is completely symmetrical. The chart section showcases the confusion matrix and the about data section showcases the findings about data and format of data.")
# loading the data
df = pd.read_excel('customer_churn_ld.xlsx')
df3 = df

st.sidebar.markdown(
  '''
  # Things to note:
  - It is to be noted that finding churn with symmetrical data is only as accurate as flipping a coin
  - Data is completely symmetrical and has 50% divide between all the values.
  - Multiple classification as well as regression models are used and all gave the accuracy of around 50% always because of symmetrical data.
  - RandomForestClassifier is the model used which is displaying the chart in the chart section.
  '''
)
def about_data(df):
  st.markdown(f'''
  ### Data Summary:
  - there are total {len(df)} of data
  - there are total {df["Churn"].value_counts()[1]} number of data with Churn value as 1
  - there are total {df["Churn"].value_counts()[0]} number of data with Churn value as 0
  - customers are from {df["Location"].unique()} locations
  - there are {df["Gender"].value_counts()["Male"]} data of Males and {df["Gender"].value_counts()["Female"]} of Females
  ''')

def data_symmetry(df):
  st.write(f" LA: {df['Location'].value_counts()['Los Angeles']:}")
  st.write(f" NY: {df['Location'].value_counts()['New York']:}")
  st.write(f" Miami: {df['Location'].value_counts()['Miami']:}")
  st.write(f" Chicago: {df['Location'].value_counts()['Chicago']:}")
  st.write(f" Houston: {df['Location'].value_counts()['Houston']:}")

# creating average monthly usage feature
df['Average_Monthly_Usage'] = df['Total_Usage_GB']/df['Subscription_Length_Months']
df.head(10)
# creating bill to usage ratio feature
df['Bill_to_usage_ratio'] = df['Monthly_Bill']/df['Total_Usage_GB']
df.head()
# creaeting subscription cost feature
df['Subscription_cost'] = df['Monthly_Bill']*df['Subscription_Length_Months']
df.head()
df = pd.get_dummies(df, columns=['Gender'])
df = pd.get_dummies(df, columns=['Location'])


# dropping unrequired columns
X = df.drop(['Churn','Name'],axis=1)
Y = df['Churn']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
df2 = pd.DataFrame()
df2['Churn_actual'] = Y_test
df2['Churn_pred'] = Y_pred
df2['CustomerID'] = X_test['CustomerID'].astype(int)

def accuracy_score_print():
  st.write(f'accuracy score : {accuracy_score(Y_test,Y_pred)}')
  st.write(f'precision score : {precision_score(Y_test,Y_pred)}')
  st.write(f'recall score : {recall_score(Y_test,Y_pred)}')
  st.write(f'f1 score : {f1_score(Y_test,Y_pred)}')


tab1, tab2 = st.tabs(["📈 Chart","❓About Data"])
with tab1:
  col1,col2 = st.columns((1,1))
  with col1:
    st.image('cm1.png')
  with col2:

    # Create a contingency table of the predicted churn values vs actual churn values
    contingency_table = pd.crosstab(df2['Churn_pred'], df2['Churn_actual'])

    # Create the heatmap
    fig, ax = plt.subplots()
    sns.heatmap(contingency_table,cmap='Greens', annot=True, ax=ax, linewidths=0.5, cbar=True)



    # Add a title and axis labels
    plt.title('Customer Churn Prediction vs Actual Values')
    plt.xlabel('Predicted Churn')
    plt.ylabel('Actual Churn')

    # Display the heatmap in Streamlit
    st.pyplot(fig)

with tab2:
  col3, col4 = st.columns((1,1))
  with col3:
    about_data(df3)
    st.subheader("Accuracy Scores")
    accuracy_score_print()
  with col4:
    st.subheader("Data Symmetry:")
    data_symmetry(df3)
    st.subheader("Extra points:")
    st.markdown('''
    - as can be seen above number of people of data from all cities are equal
    - number of females and males in every city and combined are equal
    - number of people churned and not churned are equal
    - average age in every city compared is equal
    ''')
