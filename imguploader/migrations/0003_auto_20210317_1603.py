# Generated by Django 3.1.7 on 2021-03-17 16:03

from django.db import migrations, models
import imguploader.models


class Migration(migrations.Migration):

    dependencies = [
        ('imguploader', '0002_auto_20210317_1316'),
    ]

    operations = [
        migrations.AlterField(
            model_name='imgup',
            name='image',
            field=models.ImageField(blank=True, null=True, upload_to=imguploader.models.upload_image_path),
        ),
    ]
