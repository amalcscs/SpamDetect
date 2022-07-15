import os
import random
from tkinter import S
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django. contrib import messages
from django.conf import settings
from.models import *
from datetime import datetime,date,timedelta
from django.http import HttpResponse, HttpResponseRedirect
# from Spam_Detect.settings import EMAIL_HOST_USER
# from django.core.mail import send_mail


def upload_files(request):
    
    if request.method == 'POST':
        acc = uploadfile()
        acc.files = request.FILES['file']
        acc.save()
        # msg_success = "Registration successfull"
        return render(request, 'index.html')
    return render(request, 'index.html') 


