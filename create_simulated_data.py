from glob import glob
import numpy as np
import pandas as pd
set_features = sorted(open("features.txt", "r").readlines())
num_features = len(set_features)
print("num_features: ", num_features)
X = np.diag(np.full(num_features,1))
Y = np.random.choice([0], size=(num_features,1))
print(X.shape)
print(Y.shape)
data2 = np.concatenate([X,Y], axis=1)
print("data: ", data2)

from glob import glob
import numpy as np
import pandas as pd
txt_file  = glob("outputs/*.txt")
set_features = sorted(open("features.txt", "r").readlines())
num_samples = len(txt_file)
num_features = len(set_features)
print("num_samples: ", num_samples)
print("num_features: ", num_features)
sum = 0
X = np.zeros((num_samples,num_features))
sample_idx = 0
for txt in txt_file:
    fi = open(txt)
    lst_rs = fi.readlines()
    set_rs = sorted(set(lst_rs))
    for i in range(num_features):
        if set_features[i] in set_rs:
            X[sample_idx, i] = 1
    sample_idx += 1
Y = np.random.choice([1], size=(num_samples,1))
headers = []
for feature in set_features:
    headers.append(feature.replace("\n", ""))
headers.append("label")
data = np.concatenate([X,Y], axis= 1)

full_data = np.concatenate([data,data2], axis= 0)
print(full_data)
df = pd.DataFrame(full_data)
df.sample(frac = 1)
# df.to_csv("data2.csv", header=headers, index=False)
X = df.iloc[1:, :-1].values
y = df.iloc[1:, -1].values
# print(X)
# print(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Set seed for reproducibility
np.random.seed(seed=1234)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]
classifiers = [
    KNeighborsClassifier(3),
    SVC(gamma=2, C=1),
    SVC(kernel = 'rbf', random_state = 0),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]
for name, classifier in zip(names, classifiers):
    print("Classifier: ", name)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_test, y_pred)
    # print('Confusion matrix: \n', cm)
    print("Acc:", accuracy_score(y_test,y_pred))