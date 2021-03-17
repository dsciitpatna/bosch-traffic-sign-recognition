from .models import CSV
from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect

import csv
import io
import logging
logger = logging.getLogger(__name__)

# Upload of csv file


def upload_file(req):
    template_name = "upload_csv.html"
    data = CSV.objects.all()
    context = {
        'order': 'Order of the CSV should be name, email, address, phone, profile',
    }
    if req.method == "GET":
        return render(req, template_name, context)
    try:
        csv_file = req.FILES['file']
        if not csv_file.name.endswith('.csv'):
            messages.error(request, 'THIS IS NOT A CSV FILE')
            return redirect('csvFile:upload_file')
        if csv_file.multiple_chunks():
            messages.error(request, "Uploaded file is too big (%.2f MB)." % (
                csv_file.size/(1000*1000),))
            return redirect('csvFile:upload_file')

        data_set = csv_file.read().decode('UTF-8')

        io_string = io.StringIO(data_set)
        next(io_string)
        for column in csv.reader(io_string, delimiter=',', quotechar="|"):
            _, created = CSV.objects.update_or_create(
                name=column[0],
                email=column[1],
                address=column[2],
                phone=column[3],
                profile=column[4])
        context['success'] = "Succesfully loaded the data"
    except:
        logger.error(req, "oject not found")
        context['failure'] = "Failed to load the data"
    return render(req, template_name, context)


# Export csv file
def export_file(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="users.csv"'

    writer = csv.writer(response)
    writer.writerow(['name', 'email', 'address', 'phone', 'profile'])

    data = CSV.objects.all().values_list(
        'name', 'email', 'address', 'phone', 'profile')
    for d in data:
        writer.writerow(d)

    return response

# displaying data


def show(req):
    template_name = "show.html"
    data = CSV.objects.all()
    context = {"open": "Page opened", "data": data}
    return render(req, template_name, context)
