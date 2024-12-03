from django.db import models

# Create your models here.

class Upload(models.Model):
    upload = models.FileField(upload_to='videos/')