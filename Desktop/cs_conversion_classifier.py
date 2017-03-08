import pandas as pd
import numpy as np
import sklearn
import psycopg2
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV

def run_query(query):
    conn = psycopg2.connect("dbname ='dev'host ='stats-eval.c2qnlyfqjn11.us-west-2.redshift.amazonaws.com' user ='aaron_christensen' password ='Abcd1234' port ='5439'")
    cur = conn.cursor()
    cur.execute(query)
    column_names = [desc[0] for desc in cur.description]
    data = cur.fetchall()
    df = pd.DataFrame(data=data, columns = column_names)
    cur.close()
    conn.close()
    df.to_csv('Desktop/model_phone_training.csv', index = False)

with open('Documents/cs_conversion_classifier.sql', 'r') as sql_script:
    query = sql_script.read()

if raw_input('Refresh data?: ').lower() in ('y','yes'):
    run_query(query)

df = pd.read_csv('Desktop/model_phone_training.csv')

limit = min(500000,len(df))
df = df[:limit]

x_all_raw = df[list(df.columns[:-2])]
y_all = df[list(df.columns[-1:])]
y_all = np.ravel(y_all)

def preprocess_features(X):
    outX = pd.DataFrame(index=X.index)
    for col, col_data in X.iteritems():
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        if col_data.dtype == object:
            col_data = col_data.replace(['True', 'False'], [1, 0])
        if col_data.dtype == object:
            col_data = col_data.replace([True, False], [1, 0])
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)
        outX = outX.join(col_data)
    outX = outX.fillna(value=0)
    outX = outX.astype(int)
    return outX

x_all = preprocess_features(x_all_raw)

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3,
    random_state=29)

def fit_model(clf, x_train, y_train, x_test, y_test):
    classifier = clf.__class__.__name__
    clf.fit(x_train, y_train)
    y_pred_test = clf.predict(x_test)
    acc = '{:.2f}'.format(accuracy_score(y_test, y_pred_test))
    precision, recall, f1, support = precision_recall_fscore_support(y_test,
        y_pred_test, average = 'weighted')
    return classifier, precision, recall, f1, acc

def eval_model(clf):
    for i in clf:
        classifier, precision, recall, f1, acc = fit_model(i, x_train,
            y_train, x_test, y_test)
        precision, recall, f1 = round(precision,2), round(recall,2), round(f1,2)
        table.loc[len(table)] = [classifier, precision, recall, f1, acc]
        print 'Finished: ' + i.__class__.__name__
    return table

clf = [
    KNeighborsClassifier(),
    #SVC(),
    #GaussianProcessClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier(),
    #AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
    ]

# table = pd.DataFrame(index = None, columns = ['Classifier',
#     'F1 Score', 'Accuracy',])

# table = pd.DataFrame(index = None, columns = ['Classifier',
#     'Precision', 'Recall', 'F1 Score', 'Accuracy'])
#
# print eval_model(clf)

clf = RandomForestClassifier(n_jobs=-1,random_state=5)

parameters = {'n_estimators':[3, 9],
              'max_features':[5, 10, 20]
             }
clf = GridSearchCV(clf, parameters)

clf.fit(x_train, y_train)
print clf.best_estimator_
print clf.best_score_
print clf.best_params_
