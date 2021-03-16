from django.db import models

# Create your models here.
class ImgUp(models.Model):
    image = models.ImageField(blank=True)
