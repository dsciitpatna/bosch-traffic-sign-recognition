from django.urls import path
from .  import views

urlpatterns = [
    path('upload-csv/',views.profile_upload, name='profile_upload'),
    path('export/', views.export_users_csv, name='export_users_csv'),
    
]
