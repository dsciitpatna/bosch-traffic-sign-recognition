from django.shortcuts import render, redirect
from django.views.generic import ListView, DetailView
from . import forms
from django.http import Http404

from .models import ImgUp

import logging
logger = logging.getLogger(__name__)

# Create your views here.


def image_upload(request):
    if request.method == 'POST':
        form = forms.UploadImage(request.POST, request.FILES)
        try:
            img_file = request.FILES['image']
            if (img_file.name.endswith('.jpg')) or (img_file.name.endswith('.jpeg')) or (img_file.name.endswith('.png')):
                form.save()
                return render(request, 'imguploader/imgup.html', {'result':'success'})
        except: 
            return render(request,'imguploader/imgup.html',{'error':'No File selected'})    
    else:
       form = forms.UploadImage()
    return render(request, 'imguploader/imgup.html', {'form': form})


class ImageListView(ListView):
    template_name = "imguploader/list.html"

    def get_queryset(self, *args, **kwargs):
        request = self.request
        t = ImgUp.objects.all()
        print(type(t))
        return ImgUp.objects.all()


class ImageDetailView(DetailView):
    queryset = ImgUp.objects.all()
    template_name = "imguploader/detail.html"

    def get_object(self, *args, **kwargs):
        request = self.request
        slug = self.kwargs.get('slug')
        try:
            instance = ImgUp.objects.get(slug=slug, active=True)
        except ImgUp.DoesNotExist:
            raise Http404("Not Found")
        except ImgUp.MultipleObjectsReturned:
            qs = ImgUp.objects.filter(slug=slug, active=True)
            instance = qs.first()
        except:
            raise Http404("Uhmmm  ")
        return instance
