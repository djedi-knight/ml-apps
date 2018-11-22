# Data Preprocessing

# Importing the libraries
import pandas as pd # https://pandas.pydata.org/

# Importing the dataset
dataset = pd.read_csv('Data.csv')
# Set X to all columns except the last one
X = dataset.iloc[:, :-1].values
# Set Y to the last column
y = dataset.iloc[:, -1].values

# Taking care of missing data
from sklearn.preprocessing import Imputer # https://scikit-learn.org/stable/
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Encode Country labels with value between 0 and n_classes-1
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# Create OneHotEncoder to encode Country feature
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)