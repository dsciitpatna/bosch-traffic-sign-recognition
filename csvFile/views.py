from django.shortcuts import render

# Create your views here.
from .models import CSV


def show(req):
    # p = CSV()
    # p.name = "Anam"
    # p.email = "test@qwerty.com"
    # p.address = "Google meet"
    # p.profile = "Anam is the best"
    # p.phone = "124567890"
    # p.save()
    template_name = "show.html"

    data = CSV.objects.all()
    context = {"open": "Page opened", "data": data}
    return render(req, template_name, context)
