from django.urls import re_path
from . import views
from django.conf.urls.static import static
from django.conf import settings

app_name = 'imguploader'

urlpatterns = [
    re_path(r'^$',views.image_upload,name="imageup"),
]

urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
