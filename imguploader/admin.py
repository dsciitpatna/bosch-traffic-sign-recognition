from django.contrib import admin

from .models import ImgUp


class ImgUpAdmin(admin.ModelAdmin):
    list_display = ['__str__', 'slug']

    class Meta:
        model = ImgUp


admin.site.register(ImgUp, ImgUpAdmin)
