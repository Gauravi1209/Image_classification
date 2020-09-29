#import dependencies
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
%matplotlib inline

# using pandas to read the database stored in the same folder
data = pd.read_csv('mnist_train.csv')

# viewing column heads
data.head()

# extracting data rom the dataset and view them up close
a = data.iloc[3,1:].values

# reshaping the extracted data into the reasonable size
a = a.reshape(28,28).astype('uint8')
plt.imshow(a)

# preparing the data
# separating labels and data values
df_x = data.iloc[:,1:]
df_y = data.iloc[:,0]

# creating test and train size/batches
x_train, x_test, y_train, y_test,= train_test_split(df_x, df_y, test_size = 0.2, random_state=4)

# check data
y_train.head()

# call rf classifier
rf = RandomForestClassifier(n_estimators=100)

# fit thr model
rf.fit(x_train, y_train)

# prediction on test data
pred = rf.predict(x_test)

pred

# check prediction accuracy
s = y_test.values

# calculate number of correctly predicted values
count = 0
for i in range(len(pred)):
    if pred[i]==s[i]:
        count = count+1
        
count

# total values that the prediction code was run on
len(pred)

# accuracy value
11628/12000
