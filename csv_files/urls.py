from django.urls import path
from .views import profile_upload, export_users_csv, show

urlpatterns = [
    path('upload-csv/',profile_upload, name='profile_upload'),
    path('export/', export_users_csv, name='export_users_csv'),
    path('show/', show, name ="show")
    
]
