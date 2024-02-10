
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

Insurance = pd.read_csv('insurance.csv')

Insurance.isnull().any()

#Checking for duplicates
Insurance.duplicated().any()


# ### Exploratory Data Analysis
# * Using a correlation matrix to check for correlations among the columns in the dataset
sns.heatmap(Insurance.corr())


# * The correlation matrix shows there’s little or no correlation between “age” and “charges”

# ### Checking for the distribution pattern of the “charges” column

sns.distplot(Insurance['charges'])

#Plotting a pairplot to check out the relationship that exists between one column to another
sns.pairplot(Insurance);


# ### Extracting dependent and independent variables:
# 
# * The dependent variable in this case is the “charges “ while the independent variables are the other columns.

X = Insurance.drop(columns = ["charges"])

y = Insurance["charges"]
y


# ### Splitting the dataset into test and train.
# * we split our data into “test” data and “train” data, using 80 percent to train the model and using the other 20 percent to test the model.


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 0)
X_train.head()


X_train_ = pd.get_dummies(X_train, columns=["sex", "smoker", "region"], drop_first=True)
x_test_ =  pd.get_dummies(X_test, columns=["sex", "smoker", "region"], drop_first=True)


X_train_.head()


#Building and fitting the model.
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

lm.fit(X_train_,y_train)


# ## Predicting the “test” set results.


predictions = lm.predict(x_test_)
predictions




from sklearn.metrics import r2_score
print(r2_score(y_test, predictions))

