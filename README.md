# CODTECHITSolutions-TASK1

Name: M CHATURVED TEJAS

Company: CODTECH IT SOLUTIONS

ID: CT6AI335

Domain: ARTIFICAL INTELLIGENCE

Duration: from JULY 20th, 2024 to SEPTEMBER 5th, 2024.

Mentor: MUZAMMIL AHMED

**Overview of the Code**

This code snippet is designed to perform the initial steps of preparing a dataset for machine learning tasks in a Google Colab environment. It includes mounting Google Drive to access the dataset, loading the data, preprocessing it, and splitting it into training and testing sets.

**Step-by-Step Breakdown**

**Mount Google Drive:**

from google.colab import drive
drive.mount('/content/drive')
This code mounts Google Drive to the Colab environment, allowing you to access files stored in your Drive. The dataset (RawData.csv) should be located in a directory accessible via this mounted path.

**Import Libraries:**

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
**The necessary libraries are imported:**
pandas for data manipulation.
LabelEncoder and StandardScaler from sklearn for encoding categorical variables and scaling features, respectively.
train_test_split for splitting the data into training and testing sets.
**Load the Data:**
df = pd.read_csv('RawData.csv')
The dataset is loaded from a CSV file into a Pandas DataFrame named df.

**Encode Categorical Variables:**

le = LabelEncoder()
df['category'] = le.fit_transform(df['category'])
This step converts categorical data in the category column into numeric format using LabelEncoder. The fit_transform method assigns a unique integer to each category in the column.

**Split Data into Features and Target:**

X = df.drop('target', axis=1)
y = df['target']
The dataset is split into features (X) and the target variable (y). The target column is assumed to be the column you're trying to predict.

**Split Data into Training and Testing Sets:***

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
The dataset is further split into training and testing sets using an 80-20 split. The random_state=42 ensures reproducibility.

**Feature Scaling:**

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

Feature scaling is applied to standardize the features in the dataset. This step scales the features so that they have a mean of 0 and a standard deviation of 1, which can improve the performance of certain machine learning algorithms.

**Output**

![Screenshot 2024-09-04 172023](https://github.com/user-attachments/assets/bcfb0652-0923-4d02-9b77-97a974f9d1fd)
