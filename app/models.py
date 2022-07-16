from xmlrpc.client import boolean
from django.contrib.auth.models import User
from django.db import models

# Create your models here.

class uploadfile(models.Model):
    files=models.FileField(upload_to = '', null=True, blank=True)

    def __str__(self):
        return self.files
