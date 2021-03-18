from django.urls import path
from .views import image_upload, ImageListView, ImageDetailView
from django.conf.urls.static import static
from django.conf import settings


app_name = 'imguploader'

urlpatterns = [
    path('', ImageListView.as_view(), name="image_list"),
    path('upload/', image_upload, name="imageup"),
    path('view/<slug:slug>/', ImageDetailView.as_view(), name="view_image"),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
