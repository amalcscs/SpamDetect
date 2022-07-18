import os
import random
from tkinter import S
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django. contrib import messages
from django.conf import settings

from Spam_Detect.settings import MEDIA_ROOT, MEDIA_URL
from.models import *
from datetime import datetime,date,timedelta
from django.http import HttpResponse, HttpResponseRedirect
# from Spam_Detect.settings import EMAIL_HOST_USER
# from django.core.mail import send_mail
import numpy as np
import csv
import os
import glob
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
from tabulate import tabulate
# from google.colab import files
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix

def index(request):
    return render(request, 'index.html')

def output(request):
    return render(request, 'output.html')

def upload_files(request):
    if request.method == 'POST':
        acc1 = uploadfile()
        acc1.files = request.FILES['file']
        acc1.save()
        # return render(request, 'index.html')
        upfile=uploadfile.objects.filter(id=acc1.id).values('files').first()['files']
        data=pd.read_csv(MEDIA_ROOT+'\\'+upfile)
        print(data)
        df = pd.DataFrame(data)
        df.columns = list(map(str.lower, data.columns.astype(str)))
        target_col = ' label'
        df1=df
        non_floats = []
        #cleaning data from columns which have a non int/float type
        for col in df:
            if df[col].dtypes != "float64" and df[col].dtypes != "int64" and col != target_col:
                non_floats.append(col)
            elif df[col].dtypes == "int64":
                df[col] = df[col].astype(float)
        df = df.drop(columns=non_floats)
        df.info()
        print(df.shape)
        a = df[target_col].unique()
        #cleaning data from nan and infinite values
        df =df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

        #assigning numerical values to our target column (multiclass classification)
        df.loc[df[target_col] == 'DrDoS_UDP', target_col] = 0
        df.loc[df[target_col] == 'Syn', target_col] = 1
        df.loc[df[target_col] == 'BENIGN', target_col] = 2
        df.loc[df[target_col] == 'DrDoS_SSDP', target_col] = 3

        X = df.iloc[:, 0:80]
        y = df.iloc[:, 80]

        selector = SelectKBest(f_classif, k = 40)
        X_new = selector.fit_transform(X, y)

        names = X.columns.values[selector.get_support()]
        scores = selector.scores_[selector.get_support()]
        names_scores = list(zip(names, scores))
        ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'F_Scores'])
        #Sort the dataframe for better visualization
        ns_df_sorted = ns_df.sort_values(['F_Scores', 'Feat_names'], ascending = [False, True])
        print(ns_df_sorted)
        ns_df_sorted.to_excel('multiKbest.xlsx')

        clf = RandomForestClassifier()
        rfe = RFE(clf, n_features_to_select=40)
        y = y.astype('int')
        rfe.fit(X, y)

        names = X.columns.values[rfe.get_support()]
        scores = rfe.ranking_[rfe.get_support()]
        names_scores = list(zip(names, scores))
        ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'F_Scores'])
        #Sort the dataframe for better visualization
        ns_df_sorted = ns_df.sort_values(['F_Scores', 'Feat_names'], ascending = [False, True])
        print(ns_df_sorted)
        ns_df_sorted.to_excel('multiRFE.xlsx')

        names = X.columns.values[rfe.get_support()]
        scores = rfe.estimator_.feature_importances_
        names_scores = list(zip(names, scores))
        ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'F_Scores'])
        #Sort the dataframe for better visualization
        ns_df_sorted = ns_df.sort_values(['F_Scores', 'Feat_names'], ascending = [False, True])
        print(ns_df_sorted)
        ns_df_sorted.to_excel('multiRFEscore.xlsx')

        pd.DataFrame(rfe.support_,index=X.columns,columns=['important'])

        RF_model = RandomForestClassifier()
        RF_model.fit(X,y)

        names = X.columns.values[rfe.get_support()]
        scores = RF_model.feature_importances_
        names_scores = list(zip(names, scores))
        ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'F_Scores'])
        #Sort the dataframe for better visualization
        ns_df_sorted = ns_df.sort_values(['F_Scores', 'Feat_names'], ascending = [False, True])
        print(ns_df_sorted)
        ns_df_sorted.to_excel('multiRF.xlsx')

        DT_clf = DecisionTreeClassifier()
        rfe = RFE(DT_clf, n_features_to_select=40)
        y = y.astype('int')
        rfe.fit(X, y)

        names = X.columns.values[rfe.get_support()]
        scores = rfe.ranking_[rfe.get_support()]
        names_scores = list(zip(names, scores))
        ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'F_Scores'])
        #Sort the dataframe for better visualization
        ns_df_sorted = ns_df.sort_values(['F_Scores', 'Feat_names'], ascending = [False, True])
        print(ns_df_sorted)
        ns_df_sorted.to_excel('multiRFE_DT.xlsx')

        names = X.columns.values[rfe.get_support()]
        scores = rfe.estimator_.feature_importances_
        names_scores = list(zip(names, scores))
        ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'F_Scores'])
        #Sort the dataframe for better visualization
        ns_df_sorted = ns_df.sort_values(['F_Scores', 'Feat_names'], ascending = [False, True])
        print(ns_df_sorted)
        ns_df_sorted.to_excel('multiRFE_DT_score.xlsx')

        #features kept by the Random forest features selection
        colums_to_keep = ['flow packets/s', 'flow iat mean', 'flow bytes/s', 'fwd packet length min', 'init_win_bytes_backward', 'fwd iat std', 'flow duration',
        'inbound', 'average packet size', 'destination port', 'fwd packet length mean', 'min_seg_size_forward', 'fwd iat max',
        'packet length std', 'bwd iat max', 'bwd packet length max', 'packet length variance', 'total backward packets',
        'bwd iat mean','fwd iat total', 'bwd iat total', 'total length of fwd packets', 'flow iat max', 'max packet length', 
        'flow iat std', 'fwd packets/s', 'fwd packet length max', 'bwd packets/s', 'bwd header length', 'packet length mean',
        'flow iat min', 'urg flag count', 'fwd iat mean', 'min packet length', 'source port', 'avg fwd segment size', 'subflow fwd packets',
        'init_win_bytes_forward', 'subflow bwd packets', 'subflow fwd bytes', ' label']

        df_filtered = df.reindex(columns = colums_to_keep)
        df_filtered[' label'] = df_filtered[' label'].astype(int)
        print(df_filtered.shape)
        print(df_filtered.dtypes)
        df_filtered.head() 
        df_filtered.isnull().sum()
        df_filtered=df_filtered.fillna(0)
        X = df_filtered.iloc[:, 0:40]
        y = df_filtered.iloc[:, 40]

        #decision Tree
        k = 5
        kf = KFold(n_splits=k, random_state=None)
        model = DecisionTreeClassifier()
        acc_score = []
        f1_sc = []
        precision_sc = []
        recall_sc = [] 
        n = 1
        for train_index , test_index in kf.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
            X_train , X_test = X.iloc[train_index,0:40],X.iloc[test_index,0:40]
            y_train , y_test = y.iloc[train_index], y.iloc[test_index]   
            model.fit(X_train,y_train)
            pred_values = model.predict(X_test)     
            acc = accuracy_score(pred_values , y_test)  #accuracy
            precision=precision_score(y_test,pred_values, average='macro')
            recall=recall_score(y_test,pred_values, average='macro')
            fscore=f1_score(y_test,pred_values, average='macro')
            #print(classification_report(y_test, pred_values, digits=3, target_names= target_N))")
            print("Decision Tree")

            print ('Results for fold #', n)
            print ('Precision :', precision)
            print ('Recall    :',  recall)
            print ('F-score   :',  fscore)
            print ('accuracy  :',  acc, '\n')
            n = n + 1
            acc_score.append(acc)
            f1_sc.append(fscore)
            recall_sc.append(recall)
        precision_sc.append(precision)   
        avg_acc_score = sum(acc_score)/k
        avg_f1_sc = sum(f1_sc)/k
        avg_recall_sc = sum(recall_sc)/k
        avg_precision_sc = sum(precision_sc)/k
        print('\nAvg precision : {}\n'.format(avg_precision_sc))
        print('Avg recall : {}\n'.format(avg_recall_sc))
        print('Avg f1 score : {}\n'.format(avg_f1_sc))
        print('Avg accuracy : {}\n'.format(avg_acc_score))
        Dtree=avg_acc_score
        cm=confusion_matrix(pred_values,y_test)
        cm
        plt.figure(figsize=(4,4))
        sn.heatmap(cm,annot=True)
        plt.xlabel('predicted')
        plt.ylabel('Truth')

        #Random Forest
        k = 5
        kf = KFold(n_splits=k, random_state=None)
        model = RandomForestClassifier()
        acc_score = []
        f1_sc = []
        precision_sc = []
        recall_sc = [] 
        n = 1
        for train_index , test_index in kf.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
            X_train , X_test = X.iloc[train_index,0:40],X.iloc[test_index,0:40]
            y_train , y_test = y.iloc[train_index], y.iloc[test_index]   
            model.fit(X_train,y_train)
            pred_values = model.predict(X_test)   
            acc = accuracy_score(pred_values , y_test)  #accuracy
            precision=precision_score(y_test,pred_values, average='macro')
            recall=recall_score(y_test,pred_values, average='macro')
            fscore=f1_score(y_test,pred_values, average='macro')
            #print(classification_report(y_test, pred_values, digits=3, target_names= target_N))
            print("Random Forest")

            print ('Results for fold #', n)
            print ('Precision :', precision)
            print ('Recall    :',  recall)
            print ('F-score   :',  fscore)
            print ('accuracy  :',  acc, '\n')
            n = n + 1
            acc_score.append(acc)
            f1_sc.append(fscore)
            recall_sc.append(recall)
            precision_sc.append(precision)   
        avg_acc_score = sum(acc_score)/k
        avg_f1_sc = sum(f1_sc)/k
        avg_recall_sc = sum(recall_sc)/k
        avg_precision_sc = sum(precision_sc)/k
        print('\nAvg precision : {}\n'.format(avg_precision_sc))
        print('Avg recall : {}\n'.format(avg_recall_sc))
        print('Avg f1 score : {}\n'.format(avg_f1_sc))
        print('Avg accuracy : {}\n'.format(avg_acc_score))
        RF=avg_acc_score
        cm=confusion_matrix(pred_values,y_test)
        cm
        plt.figure(figsize=(4,4))
        sn.heatmap(cm,annot=True)
        plt.xlabel('predicted')
        plt.ylabel('Truth')

        #Naive Bayes
        k = 5
        kf = KFold(n_splits=k, random_state=None)
        model = GaussianNB()
        acc_score = []
        f1_sc = []
        precision_sc = []
        recall_sc = [] 
        n = 1
        for train_index , test_index in kf.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
            X_train , X_test = X.iloc[train_index,0:40],X.iloc[test_index,0:40]
            y_train , y_test = y.iloc[train_index], y.iloc[test_index]   
            model.fit(X_train,y_train)
            pred_values = model.predict(X_test)   
            acc = accuracy_score(pred_values , y_test)  #accuracy
            precision=precision_score(y_test,pred_values, average='macro')
            recall=recall_score(y_test,pred_values, average='macro')
            fscore=f1_score(y_test,pred_values, average='macro')
            #print(classification_report(y_test, pred_values, digits=3, target_names= target_N))
            print("Naive Bayes")

            print ('Results for fold #', n)
            print ('Precision :', precision)
            print ('Recall    :',  recall)
            print ('F-score   :',  fscore)
            print ('accuracy  :',  acc, '\n')
            n = n + 1
            acc_score.append(acc)
            f1_sc.append(fscore)
            recall_sc.append(recall)
            precision_sc.append(precision)   
        avg_acc_score = sum(acc_score)/k
        avg_f1_sc = sum(f1_sc)/k
        avg_recall_sc = sum(recall_sc)/k
        avg_precision_sc = sum(precision_sc)/k
        print('\nAvg precision : {}\n'.format(avg_precision_sc))
        print('Avg recall : {}\n'.format(avg_recall_sc))
        print('Avg f1 score : {}\n'.format(avg_f1_sc))
        print('Avg accuracy : {}\n'.format(avg_acc_score))
        NB=avg_acc_score
        cm=confusion_matrix(pred_values,y_test)
        cm
        plt.figure(figsize=(4,4))
        sn.heatmap(cm,annot=True)
        plt.xlabel('predicted')
        plt.ylabel('Truth')

        #KNN
        k = 5
        kf = KFold(n_splits=k, random_state=None)
        model =KNeighborsClassifier(n_neighbors = 5)
        acc_score = []
        f1_sc = []
        precision_sc = []
        recall_sc = [] 
        n = 1
        for train_index , test_index in kf.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
            X_train , X_test = X.iloc[train_index,0:40],X.iloc[test_index,0:40]
            y_train , y_test = y.iloc[train_index], y.iloc[test_index]   
            model.fit(X_train,y_train)
            pred_values = model.predict(X_test)  
            acc = accuracy_score(pred_values , y_test)  #accuracy
            precision=precision_score(y_test,pred_values, average='macro')
            recall=recall_score(y_test,pred_values, average='macro')
            fscore=f1_score(y_test,pred_values, average='macro')
            #print(classification_report(y_test, pred_values, digits=3, target_names= target_N))
            print("KNN")

            print ('Results for fold #', n)
            print ('Precision :', precision)
            print ('Recall    :',  recall)
            print ('F-score   :',  fscore)
            print ('accuracy  :',  acc, '\n')
            n = n + 1
            acc_score.append(acc)
            f1_sc.append(fscore)
            recall_sc.append(recall)
            precision_sc.append(precision)   
        avg_acc_score = sum(acc_score)/k
        avg_f1_sc = sum(f1_sc)/k
        avg_recall_sc = sum(recall_sc)/k
        avg_precision_sc = sum(precision_sc)/k
        print('\nAvg precision : {}\n'.format(avg_precision_sc))
        print('Avg recall : {}\n'.format(avg_recall_sc))
        print('Avg f1 score : {}\n'.format(avg_f1_sc))
        print('Avg accuracy : {}\n'.format(avg_acc_score))
        KNN=avg_acc_score
        cm=confusion_matrix(pred_values,y_test)
        cm
        plt.figure(figsize=(4,4))
        sn.heatmap(cm,annot=True)
        plt.xlabel('predicted')
        plt.ylabel('Truth')

        #Neural Network
        k = 5
        kf = KFold(n_splits=k, random_state=None)
        model =MLPClassifier()
        acc_score = []
        f1_sc = []
        precision_sc = []
        recall_sc = [] 
        n = 1
        for train_index , test_index in kf.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
            X_train , X_test = X.iloc[train_index,0:40],X.iloc[test_index,0:40]
            y_train , y_test = y.iloc[train_index], y.iloc[test_index]     
            model.fit(X_train,y_train)
            pred_values = model.predict(X_test)    
            acc = accuracy_score(pred_values , y_test)  #accuracy
            precision=precision_score(y_test,pred_values, average='macro')
            recall=recall_score(y_test,pred_values, average='macro')
            fscore=f1_score(y_test,pred_values, average='macro')
            #print(classification_report(y_test, pred_values, digits=3, target_names= target_N))
            print("MLP")
            print ('Results for fold #', n)
            print ('Precision :', precision)
            print ('Recall    :',  recall)
            print ('F-score   :',  fscore)
            print ('accuracy  :',  acc, '\n')
            n = n + 1
            acc_score.append(acc)
            f1_sc.append(fscore)
            recall_sc.append(recall)
            precision_sc.append(precision)   
        avg_acc_score = sum(acc_score)/k
        avg_f1_sc = sum(f1_sc)/k
        avg_recall_sc = sum(recall_sc)/k
        avg_precision_sc = sum(precision_sc)/k
        print('\nAvg precision : {}\n'.format(avg_precision_sc))
        print('Avg recall : {}\n'.format(avg_recall_sc))
        print('Avg f1 score : {}\n'.format(avg_f1_sc))
        print('Avg accuracy : {}\n'.format(avg_acc_score))
        MLP=avg_acc_score
        cm=confusion_matrix(pred_values,y_test)
        cm
        plt.figure(figsize=(4,4))
        sn.heatmap(cm,annot=True)
        plt.xlabel('predicted')
        plt.ylabel('Truth')

        #Logistic Regression
        k = 5
        kf = KFold(n_splits=k, random_state=None)
        model =LogisticRegression()
        acc_score = []
        f1_sc = []
        precision_sc = []
        recall_sc = []
        n = 1
        for train_index , test_index in kf.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
            X_train , X_test = X.iloc[train_index,0:40],X.iloc[test_index,0:40]
            y_train , y_test = y.iloc[train_index], y.iloc[test_index]     
            model.fit(X_train,y_train)
            pred_values = model.predict(X_test)     
            acc = accuracy_score(pred_values , y_test)  #accuracy
            precision=precision_score(y_test,pred_values, average='macro')
            recall=recall_score(y_test,pred_values, average='macro')
            fscore=f1_score(y_test,pred_values, average='macro')
            #print(classification_report(y_test, pred_values, digits=3, target_names= target_N))
            print("Logistic Regression")

            print ('Results for fold #', n)
            print ('Precision :', precision)
            print ('Recall    :',  recall)
            print ('F-score   :',  fscore)
            print ('accuracy  :',  acc, '\n')
            n = n + 1
            acc_score.append(acc)
            f1_sc.append(fscore)
            recall_sc.append(recall)
            precision_sc.append(precision)   
        avg_acc_score = sum(acc_score)/k
        avg_f1_sc = sum(f1_sc)/k
        avg_recall_sc = sum(recall_sc)/k
        avg_precision_sc = sum(precision_sc)/k
        print('\nAvg precision : {}\n'.format(avg_precision_sc))
        print('Avg recall : {}\n'.format(avg_recall_sc))
        print('Avg f1 score : {}\n'.format(avg_f1_sc))
        print('Avg accuracy : {}\n'.format(avg_acc_score))
        lr=avg_acc_score
        cm=confusion_matrix(pred_values,y_test)

        plt.figure(figsize=(4,4))
        sn.heatmap(cm,annot=True)
        plt.xlabel('predicted')
        plt.ylabel('Truth')

        print("Accuracy for different models")
        print("Decision Tree :",Dtree)
        print("Random Forest :",RF)
        print("Naive Bayes   :",NB)
        print("KNN           :",KNN)
        print("MLP           :",MLP)
        print("Logistic Regression:",lr)
        
            # msg_success = "successfull"
        return render(request, 'output.html',{'precision':precision,'recall':recall,'fscore':fscore,'acc':acc,
        'avg_precision_sc':avg_precision_sc,'avg_recall_sc':avg_recall_sc,'avg_f1_sc':avg_f1_sc,
        'avg_acc_score':avg_acc_score,'Dtree':Dtree,'RF':RF,'NB':NB,'KNN':KNN,'MLP':MLP,'lr':lr,'n':n})
            
  