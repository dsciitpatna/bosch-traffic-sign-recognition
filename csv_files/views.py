from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect

from django.urls import reverse
# Create your views here.
import csv
import io
from .models import Profile
from django.contrib import messages
import logging
logger = logging.getLogger(__name__)
# Create your views here.
# one parameter named request


def profile_upload(request):
    # declaring template
    template = "profile_upload.html"
    data = Profile.objects.all()
 # prompt is a context variable that can have different values      depending on their context
    prompt = {
        'order': 'Order of the CSV should be name, email, address, phone, profile',
        'profiles': data
    }
    # GET request returns the value of the data with the specified key.
    if request.method == "GET":
        return render(request, template, prompt)
    try:
        csv_file = request.FILES['file']
        if not csv_file.name.endswith('.csv'):
            messages.error(request, 'THIS IS NOT A CSV FILE')
            return redirect('csv_files:profile_upload')
        if csv_file.multiple_chunks():
            messages.error(request, "Uploaded file is too big (%.2f MB)." % (
                csv_file.size/(1000*1000),))
            return redirect('csv_files:profile_upload')
            # let's check if it is a csv file
        data_set = csv_file.read().decode('UTF-8')
        # setup a stream which is when we loop through each line we are able to handle a data in a stream
        io_string = io.StringIO(data_set)
        next(io_string)
        for column in csv.reader(io_string, delimiter=',', quotechar="|"):
            _, created = Profile.objects.update_or_create(
                name=column[0],
                email=column[1],
                address=column[2],
                phone=column[3],
                profile=column[4])

    except:
        logger.error(request, "oject not found")
    context = {}
    return render(request, template, context)


def export_users_csv(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="users.csv"'

    writer = csv.writer(response)
    writer.writerow(['name', 'email', 'address', 'phone', 'profile'])

    users = Profile.objects.all().values_list(
        'name', 'email', 'address', 'phone', 'profile')
    for user in users:
        writer.writerow(user)

    return response


def show(req):
    p = Profile()
    p.name = "Anam"
    p.phone = "1234567890"
    p.email = "artist@gmail.com"
    p.address = "ewrtyuuu"
    p.profile = "ercvfb y uiy"
    p.save()
    data = Profile.objects.all()
    context = {"open": "Yes opeend", "data": []}
    print(data)
    return render(req, 'show.html', context)
