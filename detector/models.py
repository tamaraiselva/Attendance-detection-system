from django.db import models
from django.contrib.auth.models import AbstractUser
from datetime import datetime
from django.contrib.auth.models import User

class User(models.Model):
    newusername = models.CharField(max_length=255)
    id = models.AutoField(primary_key=True)


