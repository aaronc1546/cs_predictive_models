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
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import GridSearchCV

################################################################################
############################# Data extraction ##################################
################################################################################

db_creds = "dbname ='dev'host ='stats-eval.c2qnlyfqjn11.us-west-2.redshift.amazonaws.com' user ='aaron_christensen' password ='Abcd1234' port ='5439'"

#Set the file path for the SQL query
sql_file_path = 'Documents/cs_conversion_classifier.sql'
csv_file_path = 'Desktop/model_phone_training.csv'

def run_query(query):
    conn = psycopg2.connect(db_creds)
    cur = conn.cursor()
    cur.execute(query)
    column_names = [desc[0] for desc in cur.description]
    data = cur.fetchall()
    df = pd.DataFrame(data=data, columns = column_names)
    cur.close()
    conn.close()
    df.to_csv('Desktop/model_phone_training.csv', index = False)

with open(sql_file_path, 'r') as sql_script:
    query = sql_script.read()

if raw_input('Refresh data?: ').lower() in ('y','yes'):
    df = run_query(query)

df = pd.read_csv(csv_file_path)

################################################################################
############################# Data processing ##################################
################################################################################

x_all_raw = df[list(df.columns[:-2])]
y_all = np.ravel(df[list(df.columns[-1:])])

""" Replace yes/no values with 1/0, create dummy variables from columns with
multiple values """

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

""" Normalize non-boolean data """

def normalize(x):
    for i in x:
        if len(x_all[i].unique()) > 2:
            x_all[i] = np.log(x_all[i])
            x_all[i] = x_all[i].replace('-inf',0)
    x = x.fillna(value=0)
    return x

""" Split data into training/test set """

x_all = preprocess_features(x_all_raw)
x_all = normalize(x_all)

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3,
    random_state=29)

################################################################################
################################ Fit model  ####################################
################################################################################

""" Fit model, make predictions """

def fit_model(clf, x_train, y_train, x_test, y_test):
    classifier = clf.__class__.__name__
    clf.fit(x_train, y_train)
    y_pred_test = clf.predict(x_test)
    acc = '{:.2f}'.format(accuracy_score(y_test, y_pred_test))
    precision, recall, f1, support = precision_recall_fscore_support(y_test,
        y_pred_test, average = 'weighted')
    return classifier, precision, recall, f1, acc

def eval_model(clf):
    table = pd.DataFrame(index = None, columns = ['Classifier',
        'Precision', 'Recall', 'F1 Score', 'Accuracy'])
    for i in clf:
        classifier, precision, recall, f1, acc = fit_model(i, x_train,
            y_train, x_test, y_test)
        precision, recall, f1 = round(precision,2), round(recall,2), round(f1,2)
        table.loc[len(table)] = [classifier, precision, recall, f1, acc]
        print 'Finished: ' + i.__class__.__name__
    return table

################################################################################
################# Optimizing model params with GridSearchCV ####################
################################################################################

gs_clf = AdaBoostClassifier(random_state=5)
gs_parameters = {'learning_rate': [0.1,0.2,0.3],
              'n_estimators': [400,600,800]
             }

def grid_search(clf, parameters):
    clf = GridSearchCV(clf, parameters)
    clf.fit(x_train, y_train)
    y_pred_test = clf.predict(x_test)
    #print clf.best_estimator_
    print clf.best_score_
    print clf.best_params_
    print sklearn.metrics.confusion_matrix(y_test, y_pred_test)

################################################################################
########################## Feature importance plot  ############################
################################################################################

def feat_imp_chart():
    feature_importance = clf.feature_importances_
    column_names = np.asarray(list(x_train.columns.values))
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, column_names[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.gca().xaxis.grid(True)
    plt.show()

################################################################################
################################### Run it  ####################################
################################################################################

# Best model so far
clf = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=0.1, n_estimators=400, random_state=5)

clf_exp = [
    KNeighborsClassifier(),
    #SVC(), #This is super slow
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier(),
    AdaBoostClassifier(),
    #GaussianNB(), #Low accuracy
    #QuadraticDiscriminantAnalysis() #Low accuracy
    ]

print eval_model(clf_exp)

#grid_search(gs_clf, gs_parameters)
# feat_imp_chart()
