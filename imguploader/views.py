from django.shortcuts import render, redirect
from . import forms

# Create your views here.
def image_upload(request):
    if request.method == 'POST':
        form = forms.UploadImage(request.POST,request.FILES)
        if form.is_valid():
            form.save()
            return redirect('imguploader:imageup')
    else:
        form = forms.UploadImage()
    return render(request,'imguploader/imgup.html',{'form':form})
