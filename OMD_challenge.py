import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv", index_col=0)
predictions = pd.read_csv("prediction.csv", index_col=0)
descriptions = pd.read_excel("Website Descriptions.xlsx")
print(data)
print(predictions)
print(descriptions)

# Fix missing values
for index, row in data.iterrows():
    if data.loc[index, 'Diigo'] == 'Error: value not found':
        data.loc[index, 'Diigo'] = 0

# Rename mislabeled names and categories
for index, row in descriptions.iterrows():
    if descriptions.loc[index, 'Name'] == 'Word press':
        descriptions.loc[index, 'Name'] = 'Wordpress'
    if descriptions.loc[index, 'Category'] == 'Photos Sharing':
        descriptions.loc[index, 'Category'] = 'Photo Sharing'
    if descriptions.loc[index, 'Category'] == ' Video Sharing':
        descriptions.loc[index, 'Category'] = 'Video Sharing'

# Facebook has two entries, therefore we drop the incorrect one
descriptions.drop(index=80, axis=0, inplace=True)
descriptions.reset_index(drop=True, inplace=True)

# Move some entries into more appropriate categories
for index, row in descriptions.iterrows():
    if descriptions.loc[index, 'Name'] == 'Plurk' or descriptions.loc[index, 'Name'] == 'Twitter' \
            or descriptions.loc[index, 'Name'] == 'Ubertwitter':
        descriptions.loc[index, 'Category'] = 'Social Networking'
    if descriptions.loc[index, 'Name'] == 'Posterous' or descriptions.loc[index, 'Name'] == 'Tumblr':
        descriptions.loc[index, 'Category'] = 'Blog'

# Split other category by creating categories for each
for index, row in descriptions.iterrows():
    if descriptions.loc[index, 'Category'] == 'Other':
        descriptions.loc[index, 'Category'] = descriptions.loc[index, 'Name']

categories = descriptions.Category.unique()
names = descriptions.Name
cat_dict = pd.Series(descriptions.Category.values, index=descriptions.Name).to_dict()

# create columns for ad views by category
for cat_name in categories:
    data[cat_name] = 0

for name in names:
    print(name)
    for index, row in data.iterrows():
        data.loc[index, cat_dict[name]] += int(data.loc[index, name])



X1 = data[categories].values
X2 = data[names].values
y = data['Click'].values
print(X1)
print(y)

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y, test_size=0.1)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y, test_size=0.1)

error1 = []

for n in range(1, 40):
    model = KNeighborsClassifier(n_neighbors=n)
    model.fit(X_train1, y_train1)
    y_pred1 = model.predict(X_test1)

    # Calculating error for K values between 1 and 40
    error1.append(sklearn.metrics.mean_absolute_error(y_pred1, y_test1))

print(error1)

error2 = []
for n in range(1, 40):
    model = KNeighborsClassifier(n_neighbors=n)
    model.fit(X_train2, y_train2)
    y_pred2 = model.predict(X_test2)

    # Calculating error for K values between 1 and 40
    error2.append(sklearn.metrics.mean_absolute_error(y_pred2, y_test2))

print(error2)

error3 = []
iterations = 50
for i in range(iterations):
    for n in range(1, 40):
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y, test_size=0.1, random_state=i)
        model = KNeighborsClassifier(n_neighbors=n)
        model.fit(X_train1, y_train1)
        y_pred1 = model.predict(X_test1)
        if i == 0:
            error3.append(sklearn.metrics.mean_absolute_error(y_pred1, y_test1))
        else:
            error3[n-1] += sklearn.metrics.mean_absolute_error(y_pred1, y_test1)
            error4 = [(float(j)) / iterations for j in error3]

print(error4)

integer_list = [i for i in range(1, 41)]
print(len(error4))
print(len(integer_list))
plt.bar(integer_list, error4, width=0.8)
plt.show()
