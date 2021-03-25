from django.db import models

# Create your models here.


class CSV(models.Model):
    Width = models.IntegerField()
    Height= models.IntegerField()
    Roi_x1 = models.IntegerField()
    Roi_y1 = models.IntegerField()
    Roi_x2 = models.IntegerField()
    Roi_y2 = models.IntegerField()
    ClassId = models.IntegerField()
    path = models.TextField()
    Class_name = models.CharField(max_length=150)

    def __str__(self):
        return self.path
