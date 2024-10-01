# import core ds libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy import stats
from matplotlib import style
import joblib
import plotly.express as px
import sklearn

#Import Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

#Import modules to judge models
from tempfile import mkdtemp
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, roc_curve, roc_auc_score

#Import scaling modules
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
#from sklearn.cluster import AgglomerativeClustering
#from imblearn.over_sampling import SMOTE



'''
Use this line to reload the library after making changes
import ds_utils
from importlib import reload
reload(ds_utils)
'''


def eda(df):
    """
    getting some basic information about each dataframe
    shape of dataframe i.e. number of rows and columns
    total number of rows with null values
    total number of duplicates
    data types of columns

    Args:
    df (dataframe): dataframe containing the data for analysis
    """
    print()
    print(f"Rows: {df.shape[0]} \t Columns: {df.shape[1]}")
    print()
    
    print(f"Total null rows: {df.isnull().sum().sum()}")
    print(f"Percentage null rows: {round(df.isnull().sum().sum() / df.shape[0] * 100, 2)}%")
    print()
    
    print(f"Total duplicate rows: {df[df.duplicated(keep=False)].shape[0]}")
    print(f"Percentage dupe rows: {round(df[df.duplicated(keep=False)].shape[0] / df.shape[0] * 100, 2)}%")
    print()
    
    print(df.dtypes)
    print("-----\n")
    
    print()
    print("The head of the dataframe is: ")
    display(df.head(5))
    
    print()
    print("The tail of the dataframe is:")
    display(df.tail(5))
    
    print()
    print("Description of the numerical columns is as follows")
    display(df.describe())

def impute_europa(row):
    europa_initial = ['T', 'A', 'B', 'C']
    if row['Cabin'][0] in europa_initial and pd.isna(row['HomePlanet']):
        return 'Europa'
    else:
        return row['HomePlanet']

def impute_earth(row):
    earth_initial = ['G']
    if row['Cabin'][0] in earth_initial and pd.isna(row['HomePlanet']):
        return 'Earth'
    else:
        return row['HomePlanet']
    
def impute_cryo(row):
    cryo_initial = ['T']
    if row['Cabin'][0] in cryo_initial and pd.isna(row['CryoSleep']):
        return 'False'
    else:
        return row['CryoSleep']

def passengerid_new_features(df):
    
    #Splitting Group and Member values from "PassengerId" column.
    df["Group"] = df["PassengerId"].apply(lambda x: x.split("_")[0])
    df["Member"] =df["PassengerId"].apply(lambda x: x.split("_")[1])
    
    #Grouping the "Group" feature with respect to "member" feature to check which group is travelling with how many members
    x = df.groupby("Group")["Member"].count().sort_values()
    
    #Creating a set of group values which are travelling with more than 1 members.
    y = set(x[x>1].index)
    
    #Creating a new feature "Solo" which will indicate whether the person is travelling solo or not.
    df["Travelling_Solo"] = df["Group"].apply(lambda x: x not in y)
    
    #Creating a new feature "Group_size" which will indicate each group number of members.
    df["Group_Size"]=0
    for i in x.items():
        df.loc[df["Group"]==i[0],"Group_Size"]=i[1]

def cabin_side(row):
    return row['Cabin'].split('/')[-1]

def cabin_num(row):
    return row['Cabin'].split('/')[0]

def cabin_deck(row):
    return row['Cabin'].split('/')[0]

def plot_model_result(model_type, X_train, X_test, y_train, y_test):
    
    '''This function plots a graph for the most optimised hyperparameter for each specific model chosen, logistic, 
    decision tree, knearest neighbours etc.

    A range for each models specific hyperparameters is predetermined and the model then plots how well it performs on the test
    and train data. It then returns the final ran model for further optimisation if need be.
    
    Args:
    X_train: The train data containing all the independent variables
    y_train: The train data for just the dependent variable
    X_test: The test data containing all the independent variables
    y_test: The test data for just the dependent variable
    
    '''

    train_acc = []
    test_acc = []

    if model_type == "logistic":
        
        co = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
        
        for param in co:
            logreg = LogisticRegression(C = param,solver = 'lbfgs',max_iter = 10000)
            logreg.fit(X_train, y_train)

            train_acc.append(logreg.score(X_train, y_train))
            test_acc.append(logreg.score(X_test, y_test))

        plt.figure(figsize=(10,6))
        plt.plot(co,train_acc, label = 'Train')
        plt.plot(co,test_acc, label = 'Test')
        plt.legend()
        plt.xscale('log')
        plt.ylabel("Accuracy Score")
        plt.xlabel("C Value")
        plt.show()

        index_val = test_acc.index(max(test_acc))
        max_val = co[index_val]
        logreg = LogisticRegression(C = max_val, max_iter = 10000)
        logreg.fit(X_train, y_train)
        
        print("The C value which yielded the highest test accuracy was: "+str(max_val)+", I will input this in my final model:")
        print(f"Train score: {logreg.score(X_train, y_train)}")
        print(f"Test score: {logreg.score(X_test, y_test)}")

        #plot_confusion_matrix(logreg,X_test,y_test)
        #plt.show()
        
        y_test_pred = logreg.predict(X_test)

        print(classification_report(y_test,y_test_pred))
        
        return logreg 
        
    
    if model_type == "decision":
        
        dep = [i for i in range(1,10)]

        for depth in dep:
            
            dt = DecisionTreeClassifier(max_depth = depth)
            dt_fitted = dt.fit(X_train, y_train)

            train_acc.append(dt_fitted.score(X_train, y_train))
            test_acc.append(dt_fitted.score(X_test, y_test))

        plt.figure(figsize=(10,6))
        plt.plot(dep, train_acc, color = 'purple', label = 'train')
        plt.plot(dep, test_acc, color = 'green', label = 'test')
        plt.ylabel("Accuracy Score")
        plt.xlabel("Max_Depth Value")
        plt.legend()
        plt.show()

        index_val = test_acc.index(max(test_acc))
        max_val = dep[index_val]
        
        dt = DecisionTreeClassifier(max_depth = max_val)
        dt_fitted = dt.fit(X_train, y_train)
        
        print("The max depth value which yielded the highest test accuracy was: "+str(max_val)+", I will input this in my final model:")
        print(f"Train score: {dt_fitted.score(X_train, y_train)}")
        print(f"Test score: {dt_fitted.score(X_test, y_test)}")

        #plot_confusion_matrix(dt_fitted,X_test,y_test)
        #plt.show()

        y_test_pred = dt_fitted.predict(X_test)

        print(classification_report(y_test,y_test_pred))
        
        return dt_fitted


    if model_type == "knearest":
        
        neighbours = [k for k in range(1,50,2)]

        for k in neighbours:
            KNN_model = KNeighborsClassifier(n_neighbors=k)
            KNN_model.fit(X_train, y_train)
            train_acc.append(KNN_model.score(X_train, y_train))
            test_acc.append(KNN_model.score(X_test, y_test))

        plt.figure(figsize=(10,6))
        plt.plot(neighbours,train_acc, label = 'Train')
        plt.plot(neighbours,test_acc, label = 'Test')
        plt.ylabel("Accuracy Score")
        plt.xlabel("Neighbours value")
        plt.legend()
        plt.show()

        index_val = test_acc.index(max(test_acc))
        max_val = neighbours[index_val]
        KNN_model = KNeighborsClassifier(n_neighbors= max_val)
        KNN_model.fit(X_train, y_train)

        
        print("The neighbours value which yielded the highest test accuracy was: "+str(max_val)+", I will input this in my final model:")
        print(f"Train score: {KNN_model.score(X_train, y_train)}")
        print(f"Test score: {KNN_model.score(X_test, y_test)}")

        #plot_confusion_matrix(KNN_model,X_test,y_test)
        #plt.show()
        
        y_test_pred = KNN_model.predict(X_test)

        print(classification_report(y_test,y_test_pred))

        return KNN_model

    if model_type=='randomforest':
        
        estimators = [i for i in range(1,100)]

        for i in estimators:
            my_random_forest = RandomForestClassifier(n_estimators=i)
            my_random_forest.fit(X_train, y_train)

            train_acc.append(my_random_forest.score(X_train, y_train))
            test_acc.append(my_random_forest.score(X_test, y_test))

        plt.figure(figsize=(10,6))
        plt.plot(estimators,train_acc, label = 'Train')
        plt.plot(estimators,test_acc, label = 'Test')
        plt.ylabel("Accuracy Score")
        plt.xlabel("Number of estimators")
        plt.legend()
        plt.show()

        index_val = test_acc.index(max(test_acc))
        max_val = estimators[index_val]
        my_random_forest = RandomForestClassifier(n_estimators=max_val)
        my_random_forest.fit(X_train, y_train)

        
        print("The number of trees which yielded the highest test accuracy was: "+str(max_val)+", I will input this in my final model:")
        print(f"Train score: {my_random_forest.score(X_train, y_train)}")
        print(f"Test score: {my_random_forest.score(X_test, y_test)}")

        #plot_confusion_matrix(my_random_forest,X_test,y_test)
        #plt.show()
        
        y_test_pred = my_random_forest.predict(X_test)

        print(classification_report(y_test,y_test_pred))

        return my_random_forest

    if model_type=='adaboost':
        
        estimators = [i for i in range(1,100)]

        for i in estimators:
            AdaBoost_model = AdaBoostClassifier(n_estimators=i)
            AdaBoost_model.fit(X_train, y_train)

            train_acc.append(AdaBoost_model.score(X_train, y_train))
            test_acc.append(AdaBoost_model.score(X_test, y_test))

        plt.figure(figsize=(10,6))
        plt.plot(estimators,train_acc, label = 'Train')
        plt.plot(estimators,test_acc, label = 'Test')
        plt.ylabel("Accuracy Score")
        plt.xlabel("Number of estimators")
        plt.legend()
        plt.show()

        index_val = test_acc.index(max(test_acc))
        max_val = estimators[index_val]
        AdaBoost_model = AdaBoostClassifier(n_estimators=max_val)
        AdaBoost_model.fit(X_train, y_train)

        
        print("The number of trees which yielded the highest test accuracy was: "+str(max_val)+", I will input this in my final model:")
        print(f"Train score: {AdaBoost_model.score(X_train, y_train)}")
        print(f"Test score: {AdaBoost_model.score(X_test, y_test)}")

        #plot_confusion_matrix(AdaBoost_model,X_test,y_test)
        #plt.show()
        
        y_test_pred = AdaBoost_model.predict(X_test)

        print(classification_report(y_test,y_test_pred))

        return AdaBoost_model

    if model_type=='xgboost':
        
        estimators = [i for i in range(1,100)]

        for i in estimators:
            XGB_model = XGBClassifier(n_estimators=i, verbosity = 0)
            XGB_model.fit(X_train, y_train)

            train_acc.append(XGB_model.score(X_train, y_train))
            test_acc.append(XGB_model.score(X_test, y_test))

        plt.figure(figsize=(10,6))
        plt.plot(estimators,train_acc, label = 'Train')
        plt.plot(estimators,test_acc, label = 'Test')
        plt.ylabel("Accuracy Score")
        plt.xlabel("Number of estimators")
        plt.legend()
        plt.show()

        index_val = test_acc.index(max(test_acc))
        max_val = estimators[index_val]
        XGB_model = XGBClassifier(n_estimators=max_val)
        XGB_model.fit(X_train, y_train)

        
        print("The number of trees which yielded the highest test accuracy was: "+str(max_val)+", I will input this in my final model:")
        print(f"Train score: {XGB_model.score(X_train, y_train)}")
        print(f"Test score: {XGB_model.score(X_test, y_test)}")

        #plot_confusion_matrix(XGB_model,X_test,y_test)
        #plt.show()
        
        y_test_pred = XGB_model.predict(X_test)

        print(classification_report(y_test,y_test_pred))

        return XGB_model     
