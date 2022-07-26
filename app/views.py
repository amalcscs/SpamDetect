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
import pickle

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
        data=pd.read_csv(MEDIA_ROOT+'\\'+upfile, header=None)
        print(data)
        make_prediction(data)
        value=make_prediction(data)
    return render(request, 'output.html',{'value':value})
            
def make_prediction(data):
    model = pickle.load(open("c:\\users\\amal\\downloads\\RF_model.pickle.dat", "rb"))
    
    y = model.predict(data)
    if y==0:
        result = 'It is DoS'
    elif y==1:
        result = 'It is DDoS'
    elif y==2:
        result = 'It is Ransomware'
    elif y==3:
        result = 'It is Spyware'
    print(result)
    return result
    