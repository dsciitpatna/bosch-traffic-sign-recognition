from django.urls import path

from .views import show, upload_file, export_file,train_model

urlpatterns = [
    path('show/', show, name="show"),
    path('upload/', upload_file, name="upload_file"),
    path('export/', export_file, name="export_file"),
    path('train/',train_model,name="train_model"),
]
