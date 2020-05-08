# import the requrired libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

# import the raw dataset
raw_df = pd.read_csv("D:\Sandeep- DSDJ\python docs\winequality-red.csv")

# Let us know look at the few data columns
print(raw_df.head(10))

# Now, let us see the data_types of our columns
print(raw_df.describe())

# Now we have to see if we have "null" valued columns
print(raw_df.info())

# Looks like we have all non-null columns, that's great!
target_df = raw_df['quality']

# Let us know perform a few visulizations to get better understanding of the data
raw_df.hist(bins=10, figsize=(20, 20))
plt.show()

# Now, let us visualize a correlation matrix
corr_matrix = raw_df.corr()
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, linewidths=.5, ax=ax)
plt.show()

# Now let us spot some outliers in the data
stat = target_df.describe()
print(stat)
IQR = stat['75%'] - stat['25%']
upper = stat['75%'] + 1.5 * IQR
lower = stat['25%'] - 1.5 * IQR
print('The upper and lower bounds for suspected outliers are {} and {}.'.format(upper, lower))

# Now let us classify the data as "good" or "bad" by taking a range of quality measures
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
raw_df['quality'] = pd.cut(raw_df['quality'], bins=bins, labels=group_names)

# Now let us transform the quality column to newly created columns
label_quality = LabelEncoder()
raw_df['quality'] = label_quality.fit_transform(raw_df['quality'])

# Now let us look at the first few columns to see how we transformed
print(raw_df.head(10))

# Looks like we did a decent job

# Now, let us see how many calssify as good or bad
sns.countplot(raw_df['quality'])
plt.show()

# We can observe that, the count of class '0' is much higher than class '1'. This is an imbalnce dataset
# we will deal with it later in the process

# Let us now, seperate the feature and target df for our model
feature_df = raw_df.drop('quality', axis=1)
target_df = raw_df['quality']

# Building a Model
# Split feature and target df into train,test data sets
X_train, X_Test, y_train, Y_Test = train_test_split(feature_df, target_df, test_size=0.2, random_state=42)
# As we can see we have an imbalance data set, we need to do sampling inorder to overcome this

# Upsample minority class
X_train_u, y_train_u = resample(X_train[y_train == 1],
                                y_train[y_train == 1],
                                replace=True,
                                n_samples=X_train[y_train == 0].shape[0],
                                random_state=1)

X_train_u = np.concatenate((X_train[y_train == 0], X_train_u))
y_train_u = np.concatenate((y_train[y_train == 0], y_train_u))

# Upsample using SMOTE
sm = SMOTE(random_state=12)
x_train_sm, y_train_sm = sm.fit_sample(X_train, y_train)

# Downsample majority class
X_train_d, y_train_d = resample(X_train[y_train == 0],
                                y_train[y_train == 0],
                                replace=True,
                                n_samples=X_train[y_train == 1].shape[0],
                                random_state=1)
X_train_d = np.concatenate((X_train[y_train == 1], X_train_d))
y_train_d = np.concatenate((y_train[y_train == 1], y_train_d))

print("Original shape:", X_train.shape, y_train.shape)
print("Upsampled shape:", X_train_u.shape, y_train_u.shape)
print("SMOTE sample shape:", x_train_sm.shape, y_train_sm.shape)
print("Downsampled shape:", X_train_d.shape, y_train_d.shape)

# Create the Original, Upsampled, and Downsampled training sets
methods_data = {"Original": (X_train, y_train),
                "Upsampled": (X_train_u, y_train_u),
                "SMOTE": (x_train_sm, y_train_sm),
                "Downsampled": (X_train_d, y_train_d)}

# Loop through each type of training sets and apply 5-Fold CV using RandomForest Classifier
# By default in cross_val_score StratifiedCV is used
for method in methods_data.keys():
    rf_results = cross_val_score(RandomForestClassifier(), methods_data[method][0], methods_data[method][1], cv=5,
                                 scoring='f1')
    print(f"The best F1 Score for {method} data:")
    print(rf_results.mean())

cross_val_score(RandomForestClassifier(class_weight='balanced'), X_train, y_train, cv=5, scoring='f1').mean()

# Import our required models
lr = LogisticRegression()
rf = RandomForestClassifier()
# Train our model and Predict values

#LogisticRegression
lr.fit(X_train_u, y_train_u)
pred_vals = lr.predict(X_Test)
#print(pred_vals)
print(classification_report(Y_Test, pred_vals))
#We achieved 81% accuracy

#RandomForestClassifier
rf.fit(X_train_u, y_train_u)
pred_vals = rf.predict(X_Test)
#print(pred_vals)
print(classification_report(Y_Test, pred_vals))

# Kudos! we now achieved 91% accuracy

#  We achieved 91% accuracy with RFClassifier , hence it is the best of the two models

