import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import sklearn.neighbors.quad_tree
import sklearn.neighbors.typedefs
import sklearn.utils._cython_blas
import sklearn.tree._utils


# Read the files as pandas data frames
data = pd.read_csv("data.csv", index_col=0)
predictions = pd.read_csv("prediction.csv", index_col=0)
descriptions = pd.read_excel("Website Descriptions.xlsx")
print(data)

# The Diigo column has lots of entries of 'Error: value not found'. Dropping these entries would
# dramatically reduce the size of the training data so instead we replace them with the mode value i.e. 0
for index, row in data.iterrows():
    if data.loc[index, 'Diigo'] == 'Error: value not found':
        data.loc[index, 'Diigo'] = '0'

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

# Split other category by creating categories for each
# These entries aren't similar to each other and therefore shouldn't begrouped together
for index, row in descriptions.iterrows():
    if descriptions.loc[index, 'Category'] == 'Other':
        descriptions.loc[index, 'Category'] = '{} cat'.format(descriptions.loc[index, 'Name'])

# Create a dictionary that maps column names to their relevant category
categories = descriptions.Category.unique()
names = descriptions.Name.unique()
cat_dict = pd.Series(descriptions.Category.values, index=descriptions.Name).to_dict()

# Create lists for the names of the websites, the categories and one containing both
catlist = []
for a in categories:
    catlist.append(a)
namelist = []
for a in names:
    namelist.append(a)
features = catlist + namelist

# Ensure all entries are of type integer so that mathematical operators can be applied to them
d = {}
for name in names:
    d[name] = 'int64'
data = data.astype(d)

# One entry is an obvious error: The ad views for 'dmp945855405' on one website is of order 100B whereas most values
# are between 0 and 100
# We will drop this entry
data.drop(index='dmp945855405', axis=0, inplace=True)

# create column for total views
data['total'] = 0
# For each row, calculate the total ad views 
for name in names:
    data['total'] += data[name]

# create columns for ad views by category
for cat_name in categories:
    data[cat_name] = 0
# for each row calculate the ad views for each category
for name in names:
    data[cat_dict[name]] += data[name]

# For each feature, sort each entry into 8 equal sized bins
for feature in features:
    data['{} binned'.format(feature)] = pd.qcut(data[feature], q=8, precision=0, duplicates='drop', labels=False)
print(data)

# Create list of the bin names for future use
catbinlist = []
for a in categories:
    catbinlist.append('{} binned'.format(a))
namebinlist = []
for a in names:
    namebinlist.append('{} binned'.format(a))
features_binned = catbinlist + namebinlist

name_sums = []
for name in names:
    name_sums.append(sum(data[name]))

plt.barh(range(len(name_sums)), name_sums, align='center')
plt.ylabel('Websites')
plt.xlabel('Total ad views')
plt.tight_layout()
plt.savefig('name_sums.png')
plt.close()

cat_sums = []
for cat in categories:
    cat_sums.append(sum(data[cat]))

plt.barh(range(len(cat_sums)), cat_sums, align='center')
plt.ylabel('Categories')
plt.xlabel('Total ad views')
plt.tight_layout()
plt.savefig('cat_sums.png')
plt.close()
# Now that we have finished preprocessing the data we define a way of evaluating our models. We do this using
# confusion matrices.
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Create training and test data sets
train, test = train_test_split(data, test_size=0.1)  # Split the 'data' data frame into a test and train data frames
# with 9000 train entries and 1000 test entries


X_train1 = train[categories].values
X_test1 = test[categories].values
X_train2 = train[names].values
X_test2 = test[names].values
X_train3 = train[catbinlist].values
X_test3 = test[catbinlist].values
y_train = train['Click'].values
y_test = test['Click'].values

from sklearn.neighbors import KNeighborsClassifier

error3 = []
iterations = 10
N = 40
for i in range(iterations):
    print(i)
    for n in range(1, N):
        train5, test5 = train_test_split(data, test_size=0.1)
        X_train5 = train5[categories].values
        X_test5 = test5[categories].values
        y_train5 = train5['Click'].values
        y_test5 = test5['Click'].values
        model = KNeighborsClassifier(n_neighbors=n)
        model.fit(X_train5, y_train5)  # Train the model
        y_pred5 = model.predict(X_test5)
        if i == 0:
            error3.append(sklearn.metrics.mean_absolute_error(y_pred5, y_test5))
        else:
            error3[n - 1] += sklearn.metrics.mean_absolute_error(y_pred5, y_test5)
            error4 = [(float(j)) / iterations for j in error3]

integer_list = [i for i in range(1, N)]
plt.bar(integer_list, error4, width=0.8)
plt.tight_layout()
plt.xlabel('k')
plt.ylabel('Mean error')
plt.savefig('KNNaccuracy1.png')
plt.close()

min_error = 1
for k in range(len(error4)):
    if error4[k] < min_error:
        min_error = error4[k]
        n_min1 = k

print('For category views the optimum k value is {}, its average accuracy is {}%'.format(n_min1, (1 - min_error) * 100))

error4 = []
iterations = 10
N = 40
for i in range(iterations):
    print(i)
    for n in range(1, N):
        train6, test6 = train_test_split(data, test_size=0.1)
        X_train6 = train6[names].values
        X_test6 = test6[names].values
        y_train6 = train6['Click'].values
        y_test6 = test6['Click'].values
        model = KNeighborsClassifier(n_neighbors=n)
        model.fit(X_train6, y_train6)  # Train the model
        y_pred6 = model.predict(X_test6)
        if i == 0:
            error4.append(sklearn.metrics.mean_absolute_error(y_pred6, y_test6))
        else:
            error4[n - 1] += sklearn.metrics.mean_absolute_error(y_pred6, y_test6)

error5 = [(float(j)) / iterations for j in error4]

integer_list = [i for i in range(1, N)]
plt.bar(integer_list, error5, width=0.8)
plt.tight_layout()
plt.xlabel('k')
plt.ylabel('Mean error')
plt.savefig('KNNaccuracy2.png')
plt.close()

min_error = 1
for k in range(len(error4)):
    if error5[k] < min_error:
        min_error = error5[k]
        n_min2 = k

print('For individual website views the optimum k value is {}, its average accuracy is {}%'.format(n_min2, (
            1 - min_error) * 100))

model = KNeighborsClassifier(n_neighbors=n_min1)
model.fit(X_train1, y_train)  # Train the model
y_KNN1 = model.predict(X_test1)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_KNN1, labels=[0, 1])
np.set_printoptions(precision=4)

print('\n\nResults for KNN classifier using individual website views:')
print(classification_report(y_test, y_KNN1))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['0', '1'], normalize=False,
                      title='Confusion matrix KNN individual website views')
plt.tight_layout()
plt.savefig('KNNname.png')
plt.close()

model = KNeighborsClassifier(n_neighbors=n_min2)
model.fit(X_train2, y_train)
y_KNN2 = model.predict(X_test2)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_KNN2, labels=[0, 1])
np.set_printoptions(precision=4)

print('\n\nResults for KNN classifier using category views:')
print(classification_report(y_test, y_KNN2))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['0', '1'], normalize=False, title='Confusion matrix KNN category views')
plt.tight_layout()
plt.savefig('KNNcat.png')
plt.close()

from sklearn import svm

clf = svm.SVC(kernel='linear')
clf.fit(X_train1, y_train)  # Train the model

y_SVC1 = clf.predict(X_test1)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_SVC1, labels=[0, 1])
np.set_printoptions(precision=4)

print('\n\nResults for linear support vector classifier using category views:')
print(classification_report(y_test, y_SVC1))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['0', '1'], normalize=False, title='Confusion matrix SVM category views')
plt.tight_layout()
plt.savefig('SVMcat.png')
plt.close()

clf = svm.SVC(kernel='linear')
clf.fit(X_train2, y_train)  # Train the model

y_SVC2 = clf.predict(X_test2)
np.average(y_SVC2)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_SVC2, labels=[0, 1])
np.set_printoptions(precision=4)

print('\n\nResults for linear support vector classifier using individual website views:')
print(classification_report(y_test, y_SVC2))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['0', '1'], normalize=False,
                      title='Confusion matrix SVM individual website views')
plt.tight_layout()
plt.savefig('SVMname.png')
plt.close()

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, max_depth=20)
clf.fit(X_train2, y_train)  # Train the model

y_RF3 = clf.predict(X_test2)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_RF3, labels=[0, 1])
np.set_printoptions(precision=4)

print('\n\nResults for random forest classifier using individual website views:')
print(classification_report(y_test, y_RF3))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['0', '1'], normalize=False,
                      title='Confusion matrix random forest individual website views')
plt.tight_layout()
plt.savefig('RFname.png')
plt.close()

clf = RandomForestClassifier(n_estimators=100, max_depth=20)
clf.fit(X_train1, y_train)  # Train the model

y_RF1 = clf.predict(X_test1)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_RF1, labels=[0, 1])
np.set_printoptions(precision=4)

print('\n\nResults for random forest classifier using category views:')
print(classification_report(y_test, y_RF1))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['0', '1'], normalize=False,
                      title='Confusion matrix random forest category views')
plt.tight_layout()
plt.savefig('RFcat.png')
plt.close()

clf = RandomForestClassifier(n_estimators=100, max_depth=20)
clf = clf.fit(X_train3, y_train)

y_RF2 = clf.predict(X_test3)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_RF2, labels=[0, 1])
np.set_printoptions(precision=4)

print('\n\nResults for random forest classifier using website binned views:')
print(classification_report(y_test, y_RF2))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['0', '1'], normalize=False,
                      title='Confusion matrix random forest website binned views')
plt.tight_layout()
plt.savefig('RFbin.png')
plt.close()

# Linear Support Vector Classification using the website names columns produced our best result over all
# We train the same type of model using all the training data
X_train = data[names].values  # We select the names columns from the whole 'data' data frame as the X training values
y_train = data['Click'].values  # We select the Click column from the whole 'data' data frame as the y training values

X_predict = predictions[
    names].values  # We select the names columns from the 'predicyions' data frame as the X prediction values

clf = svm.SVC(kernel='linear')  # Select linear support vector classifier as the model
clf.fit(X_train, y_train)  # Train the model

y_predict = clf.predict(X_predict)  # Predict 'Click values

print(np.average(y_predict))  # Calculate mean click value to compare to training data

predictions['Click'] = y_predict  # Add 'Click' column to predictions data frame with predicted values
predictions.to_csv('prediction.csv')  # Overwrite predictions.csv

import operator

coef_list = clf.coef_.sum(axis=0)
indexed = list(enumerate(coef_list))  # attach indices to the list
top_20 = sorted(indexed, key=operator.itemgetter(1))[-20:]
top_indexes = list([i for i, v in top_20])
top_coefs = coef_list[top_indexes]
top_names = names[top_indexes]

plt.barh(range(len(top_names)), top_coefs, align='center')
plt.xlabel('Model coefficient')
plt.yticks(range(len(top_names)), top_names)
plt.title('Feature Importance - Top 20')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()
